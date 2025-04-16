import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from qlstm_pennylane import QLSTM  # Ensure correct import path for your QLSTM
from tqdm import tqdm

CONFIG = {
    'seq_len': 12,
    'batch_size': 32,
    'n_qubits': 8,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 2,
    'patience': 3,
    'min_delta': 0.0001,
    'start_epoch': 100,
    'quantum_backend': "default.qubit",
    'target_shift': 60
}

def load_data():
    file_path = '/Users/ruben/Documents/GitHub/MsCThesisRubenCuriel2024/Code/EDA/Notebooks/CompleteDatasetWithProphet.csv'
    df = pd.read_csv(file_path)

    # Recover datetime if needed
    if 'ds' not in df.columns:
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            df['ds'] = pd.to_datetime(df[date_cols[0]])
        else:
            raise ValueError("No datetime column found.")

    # Rename 'yhat' to ProphetForecast if needed
    if 'yhat' in df.columns and 'ProphetForecast' not in df.columns:
        df.rename(columns={'yhat': 'ProphetForecast'}, inplace=True)

    return df.sort_values('ds').reset_index(drop=True)

def preprocess_data(df):
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        df[col] = df[col].clip(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)).fillna(df[col].median())

    df['LoadConsumption'] = df['LoadConsumption'].shift(-CONFIG['target_shift'])
    if 'ProphetForecast' in df.columns:
        df['ProphetForecast'] = df['ProphetForecast'].shift(-CONFIG['target_shift'])

    return df.dropna(subset=['LoadConsumption'])

def create_sequences(df, feature_cols):
    X, y = [], []
    for i in range(len(df) - CONFIG['seq_len'] - 1):
        seq = df[feature_cols].iloc[i:i+CONFIG['seq_len']].values
        target_idx = i + CONFIG['seq_len']
        if seq.shape == (CONFIG['seq_len'], len(feature_cols)):
            X.append(seq)
            y.append(df['LoadConsumption'].iloc[target_idx])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class QuantumLSTMModel(nn.Module):
    def __init__(self, input_size, use_dropout=True):
        super().__init__()
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
        self.qlstm2 = QLSTM(
            input_size=30,
            hidden_size=90,
            n_qubits=CONFIG['n_qubits'],
            n_qlayers=1,
            batch_first=True,
            return_sequences=False,
            backend=CONFIG['quantum_backend']
        )
        self.fc = nn.Linear(90, 1)

    def forward(self, x):
        x, _ = self.qlstm1(x)
        x = self.dropout(x)
        x, _ = self.qlstm2(x)
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.fc(x).squeeze(-1)

def train_model(model, model_name, train_loader, test_loader):
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(CONFIG['epochs']), desc="Epochs", unit="epoch"):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = torch.sqrt(nn.MSELoss()(model(X_batch), y_batch))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        y_preds, y_trues = [], []

        with torch.no_grad():
            for X_val, y_val in test_loader:
                outputs = model(X_val)
                val_loss += torch.sqrt(nn.MSELoss()(outputs, y_val))
                y_preds.extend(outputs.cpu().numpy())
                y_trues.extend(y_val.cpu().numpy())

        val_loss /= len(test_loader)
        rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
        mape = mean_absolute_percentage_error(y_trues, y_preds) * 100

        print(f"{model_name} Epoch {epoch+1}: Val Loss (RMSE): {val_loss:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")

        if epoch >= CONFIG['start_epoch']:
            if val_loss < best_loss - CONFIG['min_delta']:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'best_{model_name}.pth')
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= CONFIG['patience']:
                print(f"{model_name} early stopping at epoch {epoch}")
                break

    print(f"{model_name} Best Validation Loss: {best_loss:.4f}")
    return best_loss

def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing data...")
    df = preprocess_data(df)

    print("Splitting into train/test...")
    train_size = int(0.1 * len(df))
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    print("Selecting features...")
    feature_cols = train_df.select_dtypes(include='number').columns.drop('LoadConsumption').tolist()
    selector = RFE(LinearRegression(), n_features_to_select=10)
    selector.fit(train_df[feature_cols], train_df['LoadConsumption'])
    feature_cols = train_df[feature_cols].columns[selector.support_].tolist()

    if 'ProphetForecast' in df.columns:
        feature_cols.append('ProphetForecast')
    else:
        raise ValueError("Missing 'ProphetForecast' column in dataset.")

    print("Scaling features...")
    scaler = MinMaxScaler().fit(train_df[feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    print("Creating sequences...")
    X_train, y_train = create_sequences(train_df, feature_cols)
    X_test, y_test = create_sequences(test_df, feature_cols)

    print("Creating data loaders...")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=CONFIG['batch_size'])

    print("Initializing Quantum LSTM model (without dropout)...")
    model = QuantumLSTMModel(input_size=len(feature_cols), use_dropout=False)

    print("Starting training...")
    train_model(model, "QLSTM", train_loader, test_loader)

if __name__ == "__main__":
    main()
