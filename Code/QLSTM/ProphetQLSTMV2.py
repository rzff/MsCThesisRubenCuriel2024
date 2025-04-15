import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from prophet import Prophet
from qlstm_pennylane import QLSTM

CONFIG = {
    'seq_len': 36,
    'batch_size': 128,    # From text
    'n_qubits': 8,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 50,         # From text
    'patience': 3,        # From text (3 epochs patience)
    'min_delta': 0.0001,  # From text
    'start_epoch': 10,    # From text (start monitoring from epoch 10)
    'quantum_backend': "default.qubit",
    'target_shift': 60
}

def load_data():
    file_path = '/Users/ruben/Documents/GitHub/MsCThesisRubenCuriel2024/Code/EDA/Notebooks/CompleteDataset.csv'
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    return df.sort_values('ds').reset_index(drop=True)

def preprocess_data(df):
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        df[col] = df[col].clip(q1-1.5*(q3-q1), q3+1.5*(q3-q1)).fillna(df[col].median())
    df['LoadConsumption'] = df['LoadConsumption'].shift(-CONFIG['target_shift'])
    return df.dropna(subset=['LoadConsumption'])

def create_sequences(df, feature_cols):
    X, y = [], []
    for i in range(len(df) - CONFIG['seq_len'] - 1):  # Fix offset
        seq = df[feature_cols].iloc[i:i+CONFIG['seq_len']].values
        target_idx = i + CONFIG['seq_len']
        if seq.shape == (CONFIG['seq_len'], len(feature_cols)):
            X.append(seq)
            y.append(df['LoadConsumption'].iloc[target_idx])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class QuantumLSTMModel(nn.Module):
    def __init__(self, input_size, use_dropout=True):
        super().__init__()
        # First QLSTM layer (30 nodes)
        self.qlstm1 = QLSTM(
            input_size=input_size,
            hidden_size=30,
            n_qubits=CONFIG['n_qubits'],
            n_qlayers=1,
            batch_first=True,
            return_sequences=True,
            backend=CONFIG['quantum_backend']
        )
        self.dropout = nn.Dropout(CONFIG['dropout_rate']) if use_dropout else nn.Identity()

        # Second QLSTM layer (90 nodes)
        self.qlstm2 = QLSTM(
            input_size=30,
            hidden_size=90,
            n_qubits=CONFIG['n_qubits'],
            n_qlayers=1,
            batch_first=True,
            return_sequences=False,  # Crucial fix
            backend=CONFIG['quantum_backend']
        )
        self.fc = nn.Linear(90, 1)

    def forward(self, x):
        x, _ = self.qlstm1(x)
        x = self.dropout(x)
        x, _ = self.qlstm2(x)

        # Als x shape [batch_size, seq_len, hidden_size], pak laatste timestep:
        if x.dim() == 3:
            x = x[:, -1, :]  # Neem alleen de laatste tijdstap

        return self.fc(x).squeeze(-1)


def main():
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)

    # Train-test split
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # Feature engineering
    feature_cols = train_df.select_dtypes(include='number').columns.drop('LoadConsumption').tolist()
    selector = RFE(LinearRegression(), n_features_to_select=10)
    selector.fit(train_df[feature_cols], train_df['LoadConsumption'])
    feature_cols = train_df[feature_cols].columns[selector.support_].tolist()

    # Add Prophet features
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    prophet_model.fit(train_df[['ds', 'LoadConsumption']].rename(columns={'LoadConsumption': 'y'}))

    for df_set in [train_df, test_df]:
        forecast = prophet_model.predict(df_set[['ds']])
        df_set['prophet_trend'] = forecast['trend'].values
        df_set['prophet_seasonal'] = (forecast['yearly'] + forecast['weekly']).values

    feature_cols += ['prophet_trend', 'prophet_seasonal']

    # Scale features
    scaler = MinMaxScaler().fit(train_df[feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Create sequences
    X_train, y_train = create_sequences(train_df, feature_cols)
    X_test, y_test = create_sequences(test_df, feature_cols)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                            batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                           batch_size=CONFIG['batch_size'])

    def train_model(model, model_name):
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        best_loss, epochs_no_improve = float('inf'), 0

        for epoch in range(CONFIG['epochs']):
            # Training
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = torch.sqrt(nn.MSELoss()(outputs, y_batch))
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in test_loader:
                    outputs = model(X_val)
                    val_loss += torch.sqrt(nn.MSELoss()(outputs, y_val))
            val_loss /= len(test_loader)

            # Early stopping logic
            if epoch >= CONFIG['start_epoch']:
                if val_loss < (best_loss - CONFIG['min_delta']):
                    best_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), f'best_{model_name}.pth')
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= CONFIG['patience']:
                    print(f"{model_name} early stopping at epoch {epoch}")
                    break

            print(f"{model_name} Epoch {epoch+1}: Val RMSE = {val_loss:.4f}")

        print(f"{model_name} Best Validation RMSE: {best_loss:.4f}")
        return best_loss

    # Train both models
    # print("Training Model 1 (with dropout)")
    # model1 = QuantumLSTMModel(input_size=len(feature_cols), use_dropout=True)
    # train_model(model1, "Model1")

    print("\nTraining Model 2 (without dropout)")
    model2 = QuantumLSTMModel(input_size=len(feature_cols), use_dropout=False)
    train_model(model2, "Model2")

if __name__ == "__main__":
    main()
