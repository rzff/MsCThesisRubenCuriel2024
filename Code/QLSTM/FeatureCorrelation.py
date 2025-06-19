#!/usr/bin/env python3
"""
FeatureCorrelation.py – produce correlation heat‑maps for full feature‑feature (including target) correlations
for each fold scheme in a single run.

Running
    python FeatureCorrelation.py
creates, for each scheme:
• CSV + per‑fold heat‑maps of the correlation matrix (features + target)
• A legend CSV mapping original feature names to wrapped labels

Heat‑maps auto‑size and wrap labels so long names don't clip.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

# ─── project utilities ────────────────────────────────────────────────────
from ProphetQLSTMV3 import (
    load_and_combine_duplicates,
    preprocess_data,
    select_mixed_features_by_corr,
    CONFIG,
    get_prophet_forecast,
)

TARGET = "LoadConsumption"
SCHEMES = {
    "4230Hours_2010_2024": dict(n_folds=13, start="2010-01-01", end="2024-01-01")}

# ─── helper functions ─────────────────────────────────────────────────────

def _get_datetime_series(df: pd.DataFrame) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(df.index):
        return pd.Series(df.index, index=df.index)
    for col in ("ds", "date", "timestamp", "datetime"):
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            return df[col]
    raise ValueError("No datetime column or index found for fold splitting.")


def split_into_folds(df: pd.DataFrame, *, n_folds: int, start: str, end: str) -> List[pd.DataFrame]:
    dt = _get_datetime_series(df)
    mask = (dt >= pd.to_datetime(start)) & (dt < pd.to_datetime(end))
    df_slice = df.loc[mask].copy().sort_index()
    fold_len = len(df_slice) // n_folds
    if fold_len < 2:
        raise ValueError("Fold length < 2; adjust date range or fold count.")
    return [df_slice.iloc[i * fold_len:(i + 1) * fold_len] for i in range(n_folds)]


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    r, _ = pearsonr(x[mask], y[mask])
    return r


def select_features(fold_df: pd.DataFrame) -> List[str]:
    climate_econ = select_mixed_features_by_corr(
        fold_df, target_col=TARGET,
        n_climate=CONFIG.get("n_climate_features", 7),
        n_econ=CONFIG.get("n_econ_features", 10),
    )
    diag = fold_df.rename(columns={"date":"ds", TARGET:"y"}).copy()
    diag["lag_1"] = diag["y"].shift(1).ffill()
    diag["rolling_24"] = diag["y"].rolling(window=24, min_periods=1).mean().ffill()
    diag = diag.dropna()
    corr_vals = diag.corr(numeric_only=True)["y"].abs().sort_values(ascending=False)
    top = [c for c in corr_vals.index if c != "y"][ : CONFIG.get("n_features_to_select",0) ]
    for lf in ("lag_1","rolling_24"):
        if lf not in top:
            top.append(lf)
    features = list(dict.fromkeys(climate_econ + top))
    return [f for f in features if f in fold_df.columns]


def compute_feature_correlation_matrix(
    fold_df: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    cols = [TARGET] + features
    df_num = fold_df[cols].apply(pd.to_numeric, errors='coerce')
    return df_num.corr()


def smart_wrap(label: str, width: int = 30, max_lines: int = 3) -> str:
    """
    Wrap `label` into lines of max `width` chars, but no more than `max_lines`.
    If it overflows, we truncate and add an ellipsis.
    """
    wrapper = textwrap.TextWrapper(width=width)
    lines = wrapper.wrap(label)
    if len(lines) > max_lines:
        # keep first (max_lines‒1) lines, then ellipsize the last
        truncated = lines[: max_lines]
        # merge any leftover text onto the last line with “…”
        remainder = " ".join(lines[max_lines:])
        truncated[-1] = truncated[-1].rstrip() + "…"
        # you could also append part of `remainder`, but ellipsis is fine
        return "\n".join(truncated)
    return "\n".join(lines)

def plot_heatmap(
    data: pd.DataFrame, scheme: str, fold_idx: int, title: str, filename: Path
) -> None:
    if data.empty:
        return

    # 1) Apply smart_wrap to index & columns
    wrapped_index = [smart_wrap(lbl) for lbl in data.index]
    wrapped_cols  = [smart_wrap(lbl) for lbl in data.columns]
    data_plot = data.copy()
    data_plot.index  = wrapped_index
    data_plot.columns = wrapped_cols

    # 2) Size figure based on number of features
    n = data_plot.shape[0]
    m = data_plot.shape[1]
    fig, ax = plt.subplots(
        figsize=( max(8, m * 0.5), max(8, n * 0.5) ),
        constrained_layout=True
    )

    # 3) Draw heatmap
    sns.heatmap(
        data_plot,
        annot=True, fmt=".2f",
        cmap="vlag", center=0,
        cbar_kws={"label":"Pearson r", "shrink":0.6},
        linewidths=0.5,
        ax=ax,
        annot_kws={"size":6}
    )

    # 4) Rotate tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=7)

    ax.set_title(f"{title} | {scheme} | Fold {fold_idx}", pad=16, fontsize=10)

    # 5) Save
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"✓ heat-map saved to {filename}")

# ─── scheme runner ────────────────────────────────────────────────────────

def run_for_scheme(name: str, cfg: dict) -> None:
    print(f"\n=== Processing scheme: {name} ===")
    out_dir = Path("output") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = preprocess_data(load_and_combine_duplicates().copy(), CONFIG)
    folds = split_into_folds(base_df, **cfg)
    print(f"✓ created {len(folds)} folds")

    for idx, fld in enumerate(folds, start=1):
        dates = _get_datetime_series(fld)
        prophet_df = get_prophet_forecast(fld.copy(), dates, CONFIG)
        fld_enriched = pd.concat([fld, prophet_df], axis=1)

        features = select_features(fld_enriched)
        corr_mat = compute_feature_correlation_matrix(fld_enriched, features)

        # save correlation CSV
        csv_path = out_dir / f"correlation_matrix_{name}_Fold-{idx}.csv"
        corr_mat.to_csv(csv_path)
        print(f"✓ saved correlation matrix to {csv_path}")

        # build and save legend mapping
        legend_map = {lbl: "\n".join(textwrap.wrap(lbl, width=20)) for lbl in corr_mat.index}
        legend_df = pd.DataFrame(
            list(legend_map.items()), columns=["original_name","wrapped_label"]
        )
        legend_path = out_dir / f"legend_{name}_Fold-{idx}.csv"
        legend_df.to_csv(legend_path, index=False)
        print(f"✓ saved legend mapping to {legend_path}")

        # apply wrapping to data
        wrapped = corr_mat.rename(index=legend_map, columns=legend_map)
        # plot heatmap
        plot_heatmap(
            wrapped,
            scheme=name,
            fold_idx=idx,
            title="Feature–Feature + Target Correlation",
            filename=out_dir / f"heatmap_full_{name}_Fold-{idx}.png",
        )

# ─── entrypoint ───────────────────────────────────────────────────────────

def main() -> None:
    for scheme, cfg in SCHEMES.items():
        run_for_scheme(scheme, cfg)
    print("\nAll fold schemes processed.")


if __name__ == "__main__":
    main()
