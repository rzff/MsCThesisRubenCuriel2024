import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from qlstm_pennylane import QLSTM
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Load and prepare dataset
df = pd.read_csv(
    '/Users/ruben/Documents/GitHub/MsCThesisRubenCuriel2024/Code/EDA/Notebooks/CompleteDataset.csv'
)
df['ds'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df = df.sort_values('ds').reset_index(drop=True)

def adjust_dates_and_shift_target(df, start_date, n_days_shift):
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    df = df[df['ds'] >= pd.to_datetime(start_date)]
    df['LoadConsumption'] = df['LoadConsumption'].shift(-n_days_shift)
    df = df.dropna(subset=['LoadConsumption'])
    return df

df_Complete = adjust_dates_and_shift_target(df, '2000-01-01', 60)

# Define features and target
all_numeric = df_Complete.select_dtypes(include='number').columns.tolist()
target_col = 'LoadConsumption'
feature_cols = [col for col in all_numeric if col != target_col]

# IQR-based imputation
for col in feature_cols + [target_col]:
    series = df_Complete[col]
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_Complete[col] = series.clip(lower=lower_bound, upper=upper_bound)
    safe_values = series[(series >= lower_bound) & (series <= upper_bound)]
    median_iqr = safe_values.median()
    df_Complete[col] = df_Complete[col].fillna(median_iqr)

# RFE Feature Selection
X_rfe = df_Complete[feature_cols]
y_rfe = df_Complete[target_col]
rfe_estimator = LinearRegression()
selector = RFE(rfe_estimator, n_features_to_select=10)
selector = selector.fit(X_rfe, y_rfe)
feature_cols = list(X_rfe.columns[selector.support_])
input_size = len(feature_cols)

# Scaling
scaler = MinMaxScaler()
df_Complete[feature_cols] = scaler.fit_transform(df_Complete[feature_cols])

# Prophet Integration
prophet_df = df_Complete[['ds', 'LoadConsumption']].rename(columns={'LoadConsumption': 'y'})
prophet = Prophet()
prophet.fit(prophet_df)
forecast = prophet.predict(df_Complete[['ds']])
df_Complete['yhat'] = forecast['yhat'].values
feature_cols.append('yhat')
# Add yhat to features if it's not already there
if 'yhat' not in feature_cols:
    feature_cols.append('yhat')

# 🔧 Recalculate input_size now that we added yhat
input_size = len(feature_cols)

# Sequence builder
def create_sequences(df, feature_cols, target_col, seq_len=6):
    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(df[feature_cols].iloc[i:i + seq_len].values)
        y.append(df[target_col].iloc[i + seq_len])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# QLSTM model
class StackedQLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size_1=30, hidden_size_2=60, dropout1=0.2, dropout2=0.3, use_dropout=True):
        super().__init__()
        self.qlstm1 = QLSTM(input_size, hidden_size_1, return_sequences=True)
        self.qlstm2 = QLSTM(hidden_size_1, hidden_size_2, return_sequences=False)
        self.output = nn.Linear(hidden_size_2, 1)
        self.dropout1 = nn.Dropout(dropout1) if use_dropout else nn.Identity()
        self.dropout2 = nn.Dropout(dropout2) if use_dropout else nn.Identity()

    def forward(self, x):
        x, _ = self.qlstm1(x)
        x = self.dropout1(x)
        x, (h_n2, _) = self.qlstm2(x)
        x = self.dropout2(h_n2)
        return self.output(x).squeeze(-1)

# Evaluation logic
def evaluate_config(config):
    X_all, y_all = create_sequences(df_Complete, feature_cols, target_col, seq_len=config['sequence_length'])
    train_size = int(0.8 * len(X_all))
    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_test, y_test = X_all[train_size:], y_all[train_size:]

    model = StackedQLSTMRegressor(
    input_size=input_size,
    hidden_size_1=config['hidden_size_1'],
    hidden_size_2=config['hidden_size_2'],
    dropout1=config['dropout1'],
    dropout2=config['dropout2'],
    use_dropout=config['use_dropout']
)


    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']) if config['optimizer'] == 'Adam' else torch.optim.RMSprop(model.parameters(), lr=config['lr'])

    def rmse_loss(pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2))

    for _ in range(10):
        model.train()
        optimizer.zero_grad()
        loss = rmse_loss(model(X_train), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()
        rmse = mean_squared_error(y_test.numpy(), preds, squared=False)
    return rmse

# Search space
search_space = {
    'optimizer': ['RMSprop', 'Adam'],
    'lr': [0.001, 0.005],
    'hidden_size_1': [30, 64],
    'hidden_size_2': [60, 128],
    'dropout1': [0.0, 0.2],
    'dropout2': [0.0, 0.3],
    'sequence_length': [6, 12],
    'use_dropout': [True]
}

# Grid search
best_config = None
best_score = float('inf')
for values in product(*search_space.values()):
    config = dict(zip(search_space.keys(), values))
    try:
        score = evaluate_config(config)
        print(f"Config: {config} → RMSE: {score:.4f}")
        if score < best_score:
            best_score = score
            best_config = config
    except Exception as e:
        print(f" Skipping config {config} due to error: {e}")

print("\n Best Configuration Found:")
print(best_config)
print(f"Best RMSE: {best_score:.4f}")
