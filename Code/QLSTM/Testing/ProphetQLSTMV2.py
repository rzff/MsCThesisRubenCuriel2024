import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from prophet import Prophet
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
from qlstm_pennylane import QLSTM
from itertools import product

CONFIG = {
    'seq_len': 48,
    'batch_size': 64,
    'n_qubits': 8,
    'dropout_rate': 0.3,
    'learning_rate': 0.005,
    'epochs': 50,
    'patience': 3,
    'min_delta': 0.00001,
    'start_epoch': 3,
    'quantum_backend': "default.qubit",
    'target_shift': 960,
    'use_dropout': True,
    'n_features_to_select': 10,
    'hidden_size1': 60,
    'hidden_size2': 120,
    'use_multiplicative_seasonality': True
}

def load_and_combine_duplicates():
    file_path = '/Users/ruben/Documents/GitHub/MsCThesisRubenCuriel2024/Code/EDA/Notebooks/CompleteDatasetWithoutProphet.csv'
    df = pd.read_csv(file_path)
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
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper).fillna(df[col].median())
    df['LoadConsumption'] = df['LoadConsumption'].shift(-config['target_shift'])
    return df

def generate_prophet_forecast(train_df, forecast_dates, config):
    prophet_train = train_df.copy()
    prophet_train = prophet_train.resample('D', on='date').mean(numeric_only=True).reset_index()
    prophet_train = prophet_train[['date', 'LoadConsumption']].rename(columns={'date': 'ds', 'LoadConsumption': 'y'})

    forecast_start = forecast_dates.min().normalize()
    forecast_end = forecast_dates.max().normalize()
    daily_forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative' if config.get("use_multiplicative_seasonality", False) else 'additive'
    )
    model.add_seasonality(name='hourly', period=24, fourier_order=5)
    model.fit(prophet_train)

    future = pd.DataFrame({'ds': daily_forecast_dates})
    forecast = model.predict(future)
    forecast = forecast[['ds', 'yhat', 'trend', 'weekly', 'yearly']]
    forecast = forecast.rename(columns={'ds': 'date'}).set_index('date')

    hourly_index = pd.date_range(start=forecast_dates.min(), end=forecast_dates.max(), freq='h')
    forecast = forecast.reindex(hourly_index.normalize(), method='ffill')
    forecast.index = hourly_index
    forecast = forecast.loc[forecast_dates]
    forecast = forecast.shift(-config['target_shift'])

    return forecast

def select_features_by_corr(train_df, target_col, k=7):
    corr = train_df.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)
    return [col for col in corr.index if col != target_col][:k]

def create_sequences(df, feature_cols, config):
    X, y = [], []
    for i in range(len(df) - config['seq_len']):
        seq = df[feature_cols].iloc[i:i+config['seq_len']].values
        target_idx = i + config['seq_len']
        if target_idx < len(df):
            X.append(seq)
            y.append(df['LoadConsumption'].iloc[target_idx])
    return torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(y)).float()

class QuantumLSTMModel(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.qlstm1 = QLSTM(input_size=input_size, hidden_size=config['hidden_size1'], n_qubits=config['n_qubits'],
                            n_qlayers=1, batch_first=True, return_sequences=True, backend=config['quantum_backend'])
        self.dropout = nn.Dropout(config['dropout_rate']) if config['use_dropout'] else nn.Identity()
        self.qlstm2 = QLSTM(input_size=config['hidden_size1'], hidden_size=config['hidden_size2'],
                            n_qubits=config['n_qubits'], n_qlayers=1, batch_first=True,
                            return_sequences=False, backend=config['quantum_backend'])
        self.fc1 = nn.Linear(config['hidden_size2'], 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.qlstm1(x)
        x = self.dropout(x)
        x, _ = self.qlstm2(x)
        if x.dim() == 3:
            x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

def train_model(model, model_name, train_loader, test_loader, config, target_scaler=None):
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    best_loss = float('inf')
    best_rmse, best_mape, best_pcc = None, None, None

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
                val_loss += torch.sqrt(nn.MSELoss()(outputs, y_val)).item()
                y_preds.extend(outputs.cpu().numpy())
                y_trues.extend(y_val.cpu().numpy())

        y_preds = np.array(y_preds)
        y_trues = np.array(y_trues)

        if target_scaler:
            y_preds = target_scaler.inverse_transform(y_preds.reshape(-1, 1)).flatten()
            y_trues = target_scaler.inverse_transform(y_trues.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
        mape = mean_absolute_percentage_error(y_trues, y_preds) * 100
        pcc, _ = pearsonr(y_trues, y_preds)

        scheduler.step(val_loss)

        if val_loss < best_loss - config['min_delta']:
            best_loss = val_loss
            best_rmse, best_mape, best_pcc = rmse, mape, pcc

        print(f"Epoch {epoch+1} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | PCC: {pcc:.3f}")

    return best_loss, best_rmse, best_mape, best_pcc

def run_single_fold(train_df, test_df, config, fold_id):
    train_df = preprocess_data(train_df.copy(), config)
    test_df = preprocess_data(test_df.copy(), config)

    train_df = train_df.dropna(subset=['LoadConsumption'])
    test_df = test_df.dropna(subset=['LoadConsumption'])

    if train_df.empty or test_df.empty:
        print(f"Skipping fold {fold_id}: no valid training or testing data.")
        return None

    forecast_dates = test_df['date'].reset_index(drop=True)
    prophet_forecast = generate_prophet_forecast(train_df, forecast_dates, config)
    forecast_cols = ['yhat', 'trend', 'weekly', 'yearly']

    test_df = test_df.set_index('date')
    test_df = test_df.merge(prophet_forecast, how='left', left_index=True, right_index=True)
    test_df[forecast_cols] = test_df[forecast_cols].fillna(0)
    test_df = test_df.reset_index()

    for col in forecast_cols:
        train_df[col] = test_df[col].mean()

    selected_features = select_features_by_corr(train_df, 'LoadConsumption', config['n_features_to_select'])
    final_features = selected_features + forecast_cols

    scaler = MinMaxScaler().fit(train_df[final_features])
    train_df[final_features] = scaler.transform(train_df[final_features])
    test_df[final_features] = scaler.transform(test_df[final_features])

    target_scaler = MinMaxScaler().fit(train_df[['LoadConsumption']])
    train_df['LoadConsumption'] = target_scaler.transform(train_df[['LoadConsumption']])
    test_df['LoadConsumption'] = target_scaler.transform(test_df[['LoadConsumption']])

    X_train, y_train = create_sequences(train_df, final_features, config)
    X_test, y_test = create_sequences(test_df, final_features, config)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Skipping fold {fold_id}: Not enough sequence data")
        return None

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config['batch_size'])

    model = QuantumLSTMModel(input_size=len(final_features), config=config)
    model_name = f"fold_{fold_id}_QLSTM"

    best_loss, best_rmse, best_mape, best_pcc = train_model(
        model, model_name, train_loader, test_loader, config, target_scaler=target_scaler
    )

    print(f"Fold {fold_id} | RMSE: {best_rmse:.2f} | MAPE: {best_mape:.2f}% | PCC: {best_pcc:.3f}")

    return {
        "fold": fold_id,
        "train_start": str(train_df['date'].min().date()),
        "train_end": str(train_df['date'].max().date()),
        "test_start": str(test_df['date'].min().date()),
        "test_end": str(test_df['date'].max().date()),
        "rmse": best_rmse,
        "mape": best_mape,
        "pcc": best_pcc
    }

def run_rolling_origin_cv(df, config, start_year, end_year, results_path="rocv_results.json"):
    all_results = []
    for year in range(start_year, end_year):
        train_start = pd.Timestamp(f"{year}-01-01")
        train_end = pd.Timestamp(f"{year + 1}-12-31")
        test_start = pd.Timestamp(f"{year + 2}-01-01")
        test_end = pd.Timestamp(f"{year + 2}-12-31")

        print(f"Fold {year - start_year + 1} | Training: {train_start.date()} to {train_end.date()} | Testing: {test_start.date()} to {test_end.date()}")

        train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
        test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"Skipping fold for year {year}: insufficient data")
            continue

        fold_result = run_single_fold(train_df, test_df, config, fold_id=year - start_year + 1)
        if fold_result:
            all_results.append(fold_result)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"ROCV completed. Results saved to: {results_path}")

if __name__ == "__main__":
    try:
        df = load_and_combine_duplicates()
        run_rolling_origin_cv(df, CONFIG, start_year=2009, end_year=2018)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"DEBUG ERROR: {e}")
