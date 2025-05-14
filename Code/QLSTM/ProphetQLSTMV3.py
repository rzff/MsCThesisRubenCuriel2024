import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import glob
import re
from prophet import Prophet
from tqdm import tqdm
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from qlstm_pennylane import QLSTM
import argparse

# Argument parser to allow fold override
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, help='Fold number to run')
args = parser.parse_args()



CONFIG = {
    'seq_len': 48,
    'batch_size': 64,
    'n_qubits': 4,
    'n_qlayers': 1,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 50,
    'patience': 5,
    'min_delta': 0.00001,
    'start_epoch': 3,
    'quantum_backend': "lightning.qubit",
    'target_shift': 1440,
    'use_dropout': True,
    'n_features_to_select': 17,
    'n_climate_features': 7,
    'n_econ_features': 10,
    'hidden_size1': 60,
    'hidden_size2': 120,
    'use_multiplicative_seasonality': True,
    'use_advanced_prophet': True,
    'warmup_epochs': 3,
    'start_fold': 1,
    'config_version': 'v2.1_pcc-aware-es',
    'run_classical_lstm': True,
    'run_quantum_lstm': True
}

# Override start_fold if fold is specified
if args.fold is not None:
    CONFIG['start_fold'] = args.fold

CLIMATE_FEATURES = [
    'DailyPrecipitation', 'MaxHourlyPrecipitation', 'HDMaxPrecipitation',
    'DailyMeanTemperature', 'HourlyMinTemperature', 'HDMinTemperature',
    'HourlyMaxTemperature', 'HDMaxTemperature',
    'DailyMeanWindspeed', 'MaxHourlyMeanWindspeed', 'HDMaxMeanWindspeed',
    'MinHourlyMeanWindspeed', 'HDMinMeanWindspeed',
    'sic', 'NAO'
]

class QuantumLSTMModel(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.qlstm = QLSTM(input_size=input_size, hidden_size=config['hidden_size1'],
                           n_qubits=config['n_qubits'], n_qlayers=1, batch_first=True,
                           return_sequences=False, backend=config['quantum_backend'])
        self.dropout = nn.Dropout(config['dropout_rate']) if config['use_dropout'] else nn.Identity()
        self.fc1 = nn.Linear(config['hidden_size1'], 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.qlstm(x)              # shape: (batch, seq_len, hidden)
        x = x[:, -1, :]                   # take last timestep
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class ClassicalLSTMModel(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.lstm = nn.LSTM(input_size, config['hidden_size1'], batch_first=True)
        self.dropout = nn.Dropout(config['dropout_rate']) if config['use_dropout'] else nn.Identity()
        self.fc1 = nn.Linear(config['hidden_size1'], 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

def load_and_combine_duplicates():
    file_path = 'CompleteDatasetHourly.csv'
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

def get_latest_checkpoint(model_name):
    checkpoint_files = glob.glob(f"{model_name}_epoch_*.pth")
    if not checkpoint_files:
        return None, 0
    latest_file = max(checkpoint_files, key=lambda x: int(re.findall(r"epoch_(\d+)", x)[0]))
    last_epoch = int(re.findall(r"epoch_(\d+)", latest_file)[0])
    return latest_file, last_epoch

def generate_prophet_forecast(train_df, forecast_dates, config):
    prophet_train = train_df[['date', 'LoadConsumption']].rename(columns={'date': 'ds', 'LoadConsumption': 'y'})
    nan_before = prophet_train.shape[0]
    prophet_train = prophet_train.dropna()
    nan_after = prophet_train.shape[0]
    print(f"Prophet training loss due to NaNs: {nan_before - nan_after} rows ({(nan_before - nan_after) / nan_before * 100:.2f}%)")

    forecast_start = forecast_dates.min()
    forecast_end = forecast_dates.max()
    extended_end = forecast_end + pd.to_timedelta(config['target_shift'], unit='h')
    future = pd.DataFrame({'ds': pd.date_range(start=forecast_start, end=extended_end, freq='h')})

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative' if config.get("use_multiplicative_seasonality", False) else 'additive'
    )
    model.add_seasonality(name='hourly', period=24, fourier_order=10)
    model.fit(prophet_train)

    forecast = model.predict(future)
    forecast = forecast[['ds', 'yhat', 'trend', 'weekly', 'yearly']]
    forecast = forecast.rename(columns={'ds': 'date'}).set_index('date')
    forecast = forecast.loc[forecast_dates]
    return forecast


def generate_prophet_forecast_with_regressors(train_df, forecast_dates, config):
    prophet_train = train_df.copy()
    prophet_train = prophet_train.rename(columns={'date': 'ds', 'LoadConsumption': 'y'})

    # Select top correlated features
    corr = prophet_train.corr(numeric_only=True)['y'].abs().sort_values(ascending=False)
    top_features = [col for col in corr.index if col != 'y'][:config['n_features_to_select']]

    # Add lag-based features derived from already shifted 'y'
    prophet_train['lag_1'] = prophet_train['y'].shift(1).ffill()
    prophet_train['rolling_24'] = prophet_train['y'].rolling(window=24, min_periods=1).mean().ffill()
    top_features += ['lag_1', 'rolling_24']

    # Reorder and drop NaNs
    prophet_train = prophet_train[['ds', 'y'] + top_features]
    nan_before = prophet_train.shape[0]
    prophet_train = prophet_train.dropna()
    nan_after = prophet_train.shape[0]
    print(f"Prophet + regressors training loss due to NaNs: {nan_before - nan_after} rows")

    # Create Prophet model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=10,
        yearly_seasonality=True,
        seasonality_mode='multiplicative' if config.get("use_multiplicative_seasonality", False) else 'additive'
    )
    model.add_seasonality(name='hourly', period=24, fourier_order=10)
    model.add_seasonality(name='monthly', period=730, fourier_order=3)

    for feature in top_features:
        model.add_regressor(feature)

    # Prepare future DataFrame with same regressors
    forecast_start = forecast_dates.min()
    forecast_end = forecast_dates.max()
    extended_end = forecast_end + pd.to_timedelta(config['target_shift'], unit='h')
    future = pd.DataFrame({'ds': pd.date_range(start=forecast_start, end=extended_end, freq='h')})

    # Recompute lag_1 and rolling_24 from 'y' (which is already shifted!)
    recent_hourly = train_df.copy()
    recent_hourly = recent_hourly.rename(columns={'date': 'ds'})
    recent_hourly['y'] = recent_hourly['LoadConsumption']  # already shifted
    recent_hourly['lag_1'] = recent_hourly['y'].shift(1).ffill()
    recent_hourly['rolling_24'] = recent_hourly['y'].rolling(window=24, min_periods=1).mean().ffill()

    for feature in top_features:
        if feature in recent_hourly.columns:
            extended = pd.concat([recent_hourly[['ds', feature]], future[['ds']]]).sort_values('ds')
            extended = extended.drop_duplicates(subset='ds', keep='last').ffill()
            extended = extended.set_index('ds').reindex(future['ds']).ffill()
            future[feature] = extended[feature].values
        else:
            print(f"[WARN] Feature {feature} missing in recent_hourly — filled with 0")
            future[feature] = 0

    future = future.ffill().bfill()
    model.fit(prophet_train)

    forecast = model.predict(future)
    pct_neg = 100.0 * (forecast['yhat'] < 0).mean()
    if pct_neg > 0:
        print(f"Warning: {pct_neg:.2f}% of Prophet predictions are negative.")

    forecast = forecast[['ds', 'yhat', 'trend', 'weekly', 'yearly']]
    forecast = forecast.rename(columns={'ds': 'date'}).set_index('date')
    forecast = forecast.loc[forecast_dates]
    return forecast




def get_prophet_forecast(train_df, forecast_dates, config):
    if config.get("use_advanced_prophet", False):
        return generate_prophet_forecast_with_regressors(train_df, forecast_dates, config)
    else:
        return generate_prophet_forecast(train_df, forecast_dates, config)

def select_mixed_features_by_corr(train_df, target_col, n_climate, n_econ):
    all_corr = train_df.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)

    climate = [f for f in CLIMATE_FEATURES if f in all_corr.index]
    econ = [f for f in all_corr.index if f not in climate and f != target_col]

    top_climate = [f for f in climate if not train_df[f].isna().all()]
    top_econ = [f for f in econ if not train_df[f].isna().all()]

    selected_climate = top_climate[:n_climate]
    selected_econ = top_econ[:n_econ]
    return selected_climate + selected_econ


def create_sequences(df, feature_cols, config):
    X, y = [], []
    for i in range(len(df) - config['seq_len']):
        seq = df[feature_cols].iloc[i:i+config['seq_len']].values
        target_idx = i + config['seq_len']
        if target_idx < len(df):
            X.append(seq)
            y.append(df['LoadConsumption'].iloc[target_idx])
    return torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(y)).float()




def train_model(train_loader, val_loader, input_size, hidden_size, device,
                prophet_preds_val, used_features, target_scaler=None,
                epochs=25, patience=5, model_class=None):


    print(f"\n Used Features in This Fold ({len(used_features)} total):")
    for feat in used_features:
        print(f" - {feat}")
    print()

    model = model_class(input_size=input_size, config=CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.MSELoss()

    # Learning rate scheduler with warm-up
    warmup_epochs = CONFIG.get("warmup_epochs", 3)
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_rmse = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    progress = trange(epochs, desc="Training", leave=True)

    for epoch in progress:
        model.train()
        batch_loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Batches", leave=False)

        for i, (batch_X, batch_y) in batch_loop:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

            # Optional: update batch bar with current loss
            batch_loop.set_postfix(loss=loss.item())

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                val_preds.append(output.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        model_preds =  val_preds

        if target_scaler is not None:
            model_preds = target_scaler.inverse_transform(model_preds.reshape(-1, 1)).flatten()
            val_targets = target_scaler.inverse_transform(val_targets.reshape(-1, 1)).flatten()

        rmse = rmse = np.sqrt(mean_squared_error(val_targets, model_preds))
        mape = mean_absolute_percentage_error(val_targets, model_preds) * 100
        pcc = np.corrcoef(val_targets, model_preds)[0, 1]

        current_lr = scheduler.get_last_lr()[0]
        progress.set_description(
            f"Epoch {epoch+1} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | PCC: {pcc:.3f} | LR: {current_lr:.6f}"
        )

        # Step the scheduler
        scheduler.step()

        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model, best_val_rmse, mape, pcc


def run_single_fold(train_df, test_df, config, fold_id):
    train_df = preprocess_data(train_df.copy(), config)
    # Add lag features used by Prophet regressors
    test_df['lag_1'] = test_df['LoadConsumption'].shift(1).bfill()
    test_df['rolling_24'] = test_df['LoadConsumption'].rolling(window=24, min_periods=1).mean()
    train_df = train_df.dropna(subset=['LoadConsumption'])
    test_df = test_df.dropna(subset=['LoadConsumption'])

    if train_df.empty or test_df.empty:
        print(f"Skipping fold {fold_id}: no valid training or testing data.")
        return None

    # STEP 1: Forecast Prophet for both train and test sets separately
    train_forecast = get_prophet_forecast(train_df, train_df['date'], config)
    test_forecast = get_prophet_forecast(train_df, test_df['date'], config)  # Prophet trained only on train_df

    forecast_cols = ['yhat', 'trend', 'weekly', 'yearly']

    # STEP 2: Merge forecasts into train and test sets
    train_df = train_df.set_index('date').merge(train_forecast, how='left', left_index=True, right_index=True).reset_index()
    test_df = test_df.set_index('date').merge(test_forecast, how='left', left_index=True, right_index=True).reset_index()

    # STEP 3: Handle potential missing values
    for col in forecast_cols:
        train_df[col] = train_df[col].ffill().bfill()
        test_df[col] = test_df[col].ffill().bfill()


    # Baseline MAPE
    try:
        y_true_baseline = test_df['LoadConsumption'].values
        prophet_yhat = test_df['yhat'].values
        baseline_mape = mean_absolute_percentage_error(y_true_baseline, prophet_yhat) * 100
        print(f" Prophet baseline MAPE: {baseline_mape:.2f}%")
    except Exception as e:
        print(f"Could not compute Prophet baseline MAPE: {e}")
        baseline_mape = None

    prophet_yhat_unscaled = test_df['yhat'].copy().values


    forecast_cols = ['yhat', 'trend', 'weekly', 'yearly']
    selected_features = select_mixed_features_by_corr(
        train_df,
        target_col='LoadConsumption',
        n_climate=config.get('n_climate_features', 3),
        n_econ=config.get('n_econ_features', 2)
    )

    final_features = list(dict.fromkeys(selected_features + forecast_cols))


    # Sanity check before scaling
    if train_df[final_features].dropna().shape[0] == 0 or test_df[final_features].dropna().shape[0] == 0:
        print(f"Fold {fold_id} skipped: insufficient non-NaN data for scaling in selected features.")
        return None

    try:
        scaler = MinMaxScaler().fit(train_df[final_features])
        train_df[final_features] = scaler.transform(train_df[final_features])
        test_df[final_features] = scaler.transform(test_df[final_features])
    except ValueError as e:
        print(f"Scaling error in fold {fold_id}: {e}")
        return None

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
    prophet_preds_val = prophet_yhat_unscaled[config['seq_len'] : config['seq_len'] + len(test_loader.dataset)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "fold": fold_id,
        "train_start": str(train_df['date'].min().date()),
        "train_end": str(train_df['date'].max().date()),
        "test_start": str(test_df['date'].min().date()),
        "test_end": str(test_df['date'].max().date()),
        "prophet_baseline_mape": baseline_mape,
        "selected_features": selected_features,
        "forecast_features": forecast_cols,
        "prophet_config": {
            "advanced": config.get("use_advanced_prophet", False),
            "multiplicative": config.get("use_multiplicative_seasonality", False),
            "n_features_used": config.get("n_features_to_select", 5)
        }
    }

    if config.get("run_classical_lstm", True):
        print(f"\n>>> Training Classical LSTM for Fold {fold_id}")
        cls_model, cls_rmse, cls_mape, cls_pcc = train_model(
            train_loader, test_loader, len(final_features), config['hidden_size1'],
            device, prophet_preds_val, final_features, target_scaler,
            epochs=config['epochs'], patience=config['patience'],
            model_class=ClassicalLSTMModel
        )
        results.update({
            "cls_rmse": cls_rmse,
            "cls_mape": cls_mape,
            "cls_pcc": cls_pcc
        })
        print(f" Classical LSTM Results | RMSE: {cls_rmse:.2f} | MAPE: {cls_mape:.2f}% | PCC: {cls_pcc:.3f}")


    if config.get("run_quantum_lstm", True):
        print(f"\n>>> Training QLSTM for Fold {fold_id}")
        qlstm_model, qlstm_rmse, qlstm_mape, qlstm_pcc = train_model(
            train_loader, test_loader, len(final_features), config['hidden_size1'],
            device, prophet_preds_val, final_features, target_scaler,
            epochs=config['epochs'], patience=config['patience'],
            model_class=QuantumLSTMModel
        )
        results.update({
            "qlstm_rmse": qlstm_rmse,
            "qlstm_mape": qlstm_mape,
            "qlstm_pcc": qlstm_pcc
        })
        print(f" Quantum LSTM Results   | RMSE: {qlstm_rmse:.2f} | MAPE: {qlstm_mape:.2f}% | PCC: {qlstm_pcc:.3f}")


    # Prepare the sliced Prophet predictions (for stacking + hybrid loss)

    used_features = final_features



    if config.get("run_classical_lstm", False):
        print(f"Fold {fold_id} | Classical LSTM | RMSE: {cls_rmse:.2f} | MAPE: {cls_mape:.2f}% | PCC: {cls_pcc:.3f}")

    if config.get("run_quantum_lstm", False):
        print(f"Fold {fold_id} | Quantum LSTM   | RMSE: {qlstm_rmse:.2f} | MAPE: {qlstm_mape:.2f}% | PCC: {qlstm_pcc:.3f}")


    # Stacking
    stacking_preds, true_targets = [], []

    if config.get("run_quantum_lstm", False):
        model_for_stack = qlstm_model
    elif config.get("run_classical_lstm", False):
        model_for_stack = cls_model
    else:
        model_for_stack = None

    if model_for_stack is not None:
        print("Started stacking")
        model_for_stack.eval()
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val = X_val.to(device)
                outputs = model_for_stack(X_val).cpu()
                stacking_preds.extend(outputs.numpy())
                true_targets.extend(y_val.numpy())

        stacking_preds = np.array(stacking_preds).reshape(-1, 1)
        true_targets = np.array(true_targets).reshape(-1, 1)
        prophet_preds = prophet_yhat_unscaled[config['seq_len']:config['seq_len'] + len(true_targets)].reshape(-1, 1)

        if target_scaler:
            stacking_preds = target_scaler.inverse_transform(stacking_preds)
            true_targets = target_scaler.inverse_transform(true_targets)

        prophet_preds = prophet_preds[:len(true_targets)]

        # --- Stacking ---
        X_stack = np.hstack([prophet_preds, stacking_preds])
        meta_model = LinearRegression().fit(X_stack, true_targets)
        y_meta_pred = meta_model.predict(X_stack)

        stack_rmse = np.sqrt(mean_squared_error(true_targets, y_meta_pred))
        stack_mape = mean_absolute_percentage_error(true_targets, y_meta_pred) * 100
        stack_pcc, _ = pearsonr(true_targets.flatten(), y_meta_pred.flatten())

        print(f"Stacked Model | RMSE: {stack_rmse:.2f} | MAPE: {stack_mape:.2f}% | PCC: {stack_pcc:.3f}")

        results.update({
            "stacked_rmse": stack_rmse,
            "stacked_mape": stack_mape,
            "stacked_pcc": stack_pcc,
        })
    else:
        print("Skipping stacking: no model enabled for stacking.")

    return {
        "fold": fold_id,
        "train_start": str(train_df['date'].min().date()),
        "train_end": str(train_df['date'].max().date()),
        "test_start": str(test_df['date'].min().date()),
        "test_end": str(test_df['date'].max().date()),
        "rmse": qlstm_rmse if config.get("run_quantum_lstm", False) else cls_rmse,
        "mape": qlstm_mape if config.get("run_quantum_lstm", False) else cls_mape,
        "pcc": qlstm_pcc if config.get("run_quantum_lstm", False) else cls_pcc,
        "prophet_baseline_mape": baseline_mape,
        "stacked_rmse": stack_rmse,
        "stacked_mape": stack_mape,
        "stacked_pcc": stack_pcc,
        "selected_features": selected_features,
        "forecast_features": forecast_cols,
        "prophet_config": {
            "advanced": config.get("use_advanced_prophet", False),
            "multiplicative": config.get("use_multiplicative_seasonality", False),
            "n_features_used": config.get("n_features_to_select", 5)
        }
    }

def save_fold_to_csv(fold_result, csv_path='rocv_results.csv'):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fold_result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(fold_result)

def run_rolling_origin_cv(df, config, start_year=2010, final_test_year=2019, results_path="rocv_results.json"):
    all_results = []
    for test_year in range(start_year + 2, final_test_year + 1):
        fold_id = test_year - (start_year + 1)
        if fold_id < config.get("start_fold", 1):
            print(f"Skipping fold {fold_id} due to config['start_fold'] = {config['start_fold']}")
            continue

        train_start = pd.Timestamp(f"{start_year}-01-01")
        train_end = pd.Timestamp(f"{test_year - 1}-12-31")
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")

        print(f"Fold {fold_id} | Training: {train_start.date()} to {train_end.date()} | Testing: {test_start.date()} to {test_end.date()}")

        train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
        test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"Skipping fold for test year {test_year}: insufficient data")
            continue

        fold_result = run_single_fold(train_df, test_df, config, fold_id=fold_id)
        if fold_result:
            all_results.append(fold_result)
            save_fold_to_csv(fold_result)

    with open(f"results/fold_{fold_result['fold']}.json", "w") as f:
        json.dump(fold_result, f, indent=4)


    print(f"ROCV completed. Results saved to: {results_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number to run")
    args = parser.parse_args()

    CONFIG['start_fold'] = args.fold  # Override start_fold to match the single fold

    df = load_and_combine_duplicates()
    df_shifted = preprocess_data(df.copy(), CONFIG)

    # Run only the specified fold
    run_rolling_origin_cv(
        df_shifted,
        CONFIG,
        start_year=2010,
        final_test_year=2010 + args.fold + 1,  # this ensures only that fold runs
        results_path=f"results1440/fold_{args.fold}_results.json"
    )
