import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
json_dir = '/Users/ruben/Documents/GitHub/MsCThesisRubenCuriel2024/Code/QLSTM/Results/jsonresult'
data_path = '/Users/ruben/Documents/GitHub/MsCThesisRubenCuriel2024/Code/QLSTM/CompleteDatasetHourly.csv'
output_dir = os.path.join(os.path.dirname(json_dir), 'plots_winsor_vs_impute')
os.makedirs(output_dir, exist_ok=True)

# === Load full dataset ===
df_full = pd.read_csv(data_path, parse_dates=['date'])
df_full = df_full.set_index('date').sort_index()

# === IQR Utilities ===
def get_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def impute_series(series, lower, upper):
    return series.apply(lambda x: lower if x < lower else (upper if x > upper else x))

def winsorize_series(series, lower, upper):
    return series.clip(lower=lower, upper=upper)

# === Main Loop ===
for filename in sorted(os.listdir(json_dir)):
    if not filename.endswith('.json'):
        continue

    with open(os.path.join(json_dir, filename), 'r') as f:
        fold_data = json.load(f)

    fold = fold_data['fold']
    train_start = pd.to_datetime(fold_data['train_start'])
    train_end = pd.to_datetime(fold_data['train_end'])
    selected_features = fold_data.get('selected_features', [])

    # Filter to training window
    df_train = df_full[(df_full.index >= train_start) & (df_full.index <= train_end)]

    # Only keep features that exist
    available_features = [feat for feat in selected_features if feat in df_train.columns]
    if not available_features:
        print(f"[Fold {fold}] No available features found.")
        continue

    n_feats = len(available_features)
    fig, axes = plt.subplots(n_feats, 1, figsize=(14, 3 * n_feats), sharex=True)

    if n_feats == 1:
        axes = [axes]

    for ax, feature in zip(axes, available_features):
        series = df_train[feature].dropna()
        if len(series) < 10:
            ax.set_title(f"{feature} (too sparse)")
            continue

        lower, upper = get_bounds(series)
        imputed = impute_series(series, lower, upper)
        winsorized = winsorize_series(series, lower, upper)

        ax.plot(series.index, series, label='Original', alpha=0.4)
        ax.plot(series.index, imputed, label='IQR Imputed', alpha=0.7)
        ax.plot(series.index, winsorized, label='Winsorized', alpha=0.7)
        ax.set_title(feature)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f'Fold {fold} – Selected Feature Comparison (Outlier Treatments)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.98, 0.98])

    plot_path = os.path.join(output_dir, f'fold_{fold}_selected_features.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"[Fold {fold}] Saved plot to: {plot_path}")
