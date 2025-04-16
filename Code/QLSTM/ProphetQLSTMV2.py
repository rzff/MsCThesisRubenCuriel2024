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
from scipy.stats import pearsonr
from qlstm_pennylane import QLSTM
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
import json
import os

CONFIG = {
    'seq_len': 12,
    'batch_size': 32,
    'n_qubits': 8,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 20,
    'patience': 3,
    'min_delta': 0.0001,
    'start_epoch': 3,
    'quantum_backend': "default.qubit",
    'target_shift': 60,
    'use_dropout': False,
    'n_features_to_select': 5
}

SEARCH_SPACE = {
    'n_features_to_select': list(range(3, 11)),
    'use_dropout': [True, False],
    'seq_len': [12,24,36],
    'batch_size': [32,64,128],
    'learning_rate': [0.001,0.005,0.01],
    'dropout_rate': [0.3,0.1,0.8]
}

def load_and_combine_duplicates():
    file_path = '/Users/ruben/Documents/GitHub/MsCThesisRubenCuriel2024/Code/EDA/Notebooks/CompleteDatasetWithProphet.csv'
    df = pd.read_csv(file_path)

    if 'date' not in df.columns:
        raise ValueError("No 'date' column found in dataset")

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.groupby(df.index).mean(numeric_only=True)
    df = df.sort_index().reset_index()
    return df

def preprocess_data(df, config):
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        df[col] = df[col].clip(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)).fillna(df[col].median())
    df['LoadConsumption'] = df['LoadConsumption'].shift(-config['target_shift'])
    if 'ProphetForecast' in df.columns:
        df['ProphetForecast'] = df['ProphetForecast'].shift(-config['target_shift'])
    return df

def create_sequences(df, feature_cols, config):
    X, y = [], []
    for i in range(len(df) - config['seq_len'] - 1):
        seq = df[feature_cols].iloc[i:i+config['seq_len']].values
        target_idx = i + config['seq_len']
        if seq.shape == (config['seq_len'], len(feature_cols)):
            X.append(seq)
            y.append(df['LoadConsumption'].iloc[target_idx])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class QuantumLSTMModel(nn.Module):
    def __init__(self, input_size, use_dropout=False, config=CONFIG):
        super().__init__()
        self.qlstm1 = QLSTM(input_size=input_size, hidden_size=30, n_qubits=config['n_qubits'],
                            n_qlayers=1, batch_first=True, return_sequences=True, backend=config['quantum_backend'])
        self.dropout = nn.Dropout(config['dropout_rate']) if use_dropout else nn.Identity()
        self.qlstm2 = QLSTM(input_size=30, hidden_size=90, n_qubits=config['n_qubits'],
                            n_qlayers=1, batch_first=True, return_sequences=False, backend=config['quantum_backend'])
        self.fc = nn.Linear(90, 1)

    def forward(self, x):
        x, _ = self.qlstm1(x)
        x = self.dropout(x)
        x, _ = self.qlstm2(x)
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.fc(x).squeeze(-1)

def train_model(model, model_name, train_loader, test_loader, config, target_scaler=None):
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    best_loss = float('inf')
    epochs_no_improve = 0
    best_rmse, best_mape, best_pcc = None, None, None
    train_losses, val_losses = [], []

    for epoch in tqdm(range(config['epochs']), desc="Epochs", unit="epoch"):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = torch.sqrt(nn.MSELoss()(model(X_batch), y_batch))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, y_preds, y_trues = 0, [], []
        with torch.no_grad():
            for X_val, y_val in test_loader:
                outputs = model(X_val)
                val_loss += torch.sqrt(nn.MSELoss()(outputs, y_val))
                y_preds.extend(outputs.cpu().numpy())
                y_trues.extend(y_val.cpu().numpy())

        val_loss /= len(test_loader)
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        y_preds = np.array(y_preds)
        y_trues = np.array(y_trues)

        if target_scaler:
            y_preds = target_scaler.inverse_transform(y_preds.reshape(-1, 1)).flatten()
            y_trues = target_scaler.inverse_transform(y_trues.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
        mape = mean_absolute_percentage_error(y_trues, y_preds) * 100
        pcc, _ = pearsonr(y_trues, y_preds)

        print(f"{model_name} Epoch {epoch+1}: Val Loss: {val_loss:.4f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | PCC: {pcc:.3f}")

        if epoch >= config['start_epoch']:
            if val_loss < best_loss - config['min_delta']:
                best_loss = val_loss
                best_rmse = rmse
                best_mape = mape
                best_pcc = pcc
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'best_{model_name}.pth')
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= config['patience']:
                print(f"{model_name} early stopping at epoch {epoch+1}")
                break

    return float(best_loss), float(best_rmse), float(best_mape), float(best_pcc), epoch+1, train_losses, val_losses, y_trues, y_preds

def main(config_overrides):
    config = CONFIG.copy()
    config.update(config_overrides)

    df = load_and_combine_duplicates()
    df = preprocess_data(df, config)
    df = df.dropna(subset=['LoadConsumption'])

    train_size = int(0.8 * len(df))
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    feature_cols = train_df.select_dtypes(include='number').columns.drop('LoadConsumption').tolist()
    selector = RFE(LinearRegression(), n_features_to_select=config['n_features_to_select'])
    selector.fit(train_df[feature_cols], train_df['LoadConsumption'])
    feature_cols = train_df[feature_cols].columns[selector.support_].tolist()
    if 'ProphetForecast' in df.columns:
        feature_cols.append('ProphetForecast')

    scaler = MinMaxScaler().fit(train_df[feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    target_scaler = MinMaxScaler().fit(train_df[['LoadConsumption']])
    train_df['LoadConsumption'] = target_scaler.transform(train_df[['LoadConsumption']])
    test_df['LoadConsumption'] = target_scaler.transform(test_df[['LoadConsumption']])

    X_train, y_train = create_sequences(train_df, feature_cols, config)
    X_test, y_test = create_sequences(test_df, feature_cols, config)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config['batch_size'])

    model = QuantumLSTMModel(input_size=len(feature_cols), use_dropout=config['use_dropout'], config=config)

    model_name = f"QLSTM_nf{config['n_features_to_select']}_do{config['use_dropout']}"
    best_loss, best_rmse, best_mape, best_pcc, best_epoch, train_losses, val_losses, y_trues, y_preds = train_model(
        model, model_name, train_loader, test_loader, config, target_scaler=target_scaler
    )

    return {
        "config": config,
        "loss": float(best_loss),
        "rmse": float(best_rmse),
        "mape": float(best_mape),
        "pcc": float(best_pcc)
    }


if __name__ == "__main__":
    combinations = list(product(*SEARCH_SPACE.values()))
    keys = list(SEARCH_SPACE.keys())

    results = []
    best_result = {"loss": float('inf')}

    os.makedirs("tuning_results", exist_ok=True)

    for i, values in enumerate(combinations):
        combo = dict(zip(keys, values))
        print(f"\n=== Running configuration {i+1}/{len(combinations)}: {combo} ===")
        try:
            result = main(combo)
            results.append(result)
            if result['loss'] < best_result['loss']:
                best_result = result
        except Exception as e:
            print(f"Run {i+1} failed with error: {e}")

    with open("tuning_results/all_results.json", "w") as f:
        json.dump(results, f, indent=4)

    with open("tuning_results/best_config.json", "w") as f:
        json.dump(best_result, f, indent=4)

    print("\nBest configuration:")
    print(json.dumps(best_result, indent=4))
