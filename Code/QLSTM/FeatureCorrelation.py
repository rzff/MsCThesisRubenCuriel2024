#!/usr/bin/env python3
"""
FeatureCorrelation.py – produce correlation heat‑maps for **both** fold
schemes in a single run (no flags needed).

Running
    python FeatureCorrelation.py
creates:
• CSV + per-fold heat‑maps for the 4-fold 2019-2024 scheme
• CSV + per-fold heat‑maps for the 8-fold 2010-2019 scheme

Heat‑maps auto‑size so axis labels never clip.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

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
PROPHET_COLS = ["yhat", "trend", "weekly", "yearly"]
SCHEMES = {
    "4folds_2019_2024": dict(n_folds=4, start="2019-01-01", end="2024-01-01"),
    "8folds_2010_2019": dict(n_folds=8, start="2010-01-01", end="2019-01-01"),
}

# ─── helper functions ─────────────────────────────────────────────────────

def _get_datetime_series(df: pd.DataFrame) -> pd.Series:
    """Return a datetime Series for the DataFrame (index or common column names)."""
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
    return [df_slice.iloc[i * fold_len : (i + 1) * fold_len] for i in range(n_folds)]


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    r, _ = pearsonr(x[mask], y[mask])
    return r


def correlations_for_fold(
    fold_df: pd.DataFrame, n_climate: int, n_econ: int
) -> List[Tuple[str, float]]:
    # ── Replicate ProphetQLSTMV3 regressor selection ──
    prophet_train = fold_df.copy().rename(columns={'date': 'ds', 'LoadConsumption': 'y'})
    prophet_train['lag_1'] = prophet_train['y'].shift(1).ffill()
    prophet_train['rolling_24'] = prophet_train['y'].rolling(window=24, min_periods=1).mean().ffill()
    before_drop = prophet_train.shape[0]
    prophet_train = prophet_train.dropna()
    after_drop = prophet_train.shape[0]
    print(f"[Diagnostic] Prophet train rows dropped due to NaNs: {before_drop - after_drop}")
    corr_vals = prophet_train.corr(numeric_only=True)['y'].abs().sort_values(ascending=False)
    n_feat = CONFIG.get('n_features_to_select', 0)
    top_regressors = [c for c in corr_vals.index if c != 'y'][:n_feat]
    for lf in ['lag_1', 'rolling_24']:
        if lf not in top_regressors:
            top_regressors.append(lf)
    print(f"[Diagnostic] selected regressors: {top_regressors}")

    # Base climate/econ feature selection
    sel = select_mixed_features_by_corr(
        fold_df,
        target_col=TARGET,
        n_climate=n_climate,
        n_econ=n_econ,
    )
    print(f"[Diagnostic] select_mixed_features_by_corr selected: {sel}")

    # Combine with Prophet regressors
    sel += [c for c in top_regressors if c in fold_df.columns and c not in sel]
    print(f"[Diagnostic] features after including regressors: {sel}")

    out: List[Tuple[str, float]] = []
    for feat in sel:
        if feat not in fold_df.columns:
            print(f"[Diagnostic] skipping {feat}: not in DataFrame")
            continue
        if fold_df[feat].dtype.kind not in "fi":
            fold_df[feat] = pd.to_numeric(fold_df[feat], errors="coerce")
        if fold_df[feat].dtype.kind not in "fi":
            print(f"[Diagnostic] skipping {feat}: non-numeric after coercion")
            continue
        r = safe_pearson(fold_df[feat].values, fold_df[TARGET].values)
        if r is None:
            print(f"[Diagnostic] insufficient finite data for {feat}, setting r=0.0")
            r = 0.0
        out.append((feat, r))
    return out


def compute_prophet_only_correlation(
    fold_df: pd.DataFrame
) -> dict[str, float]:
    """
    Compute Pearson correlations between Prophet forecast columns and the target separately.
    Returns a dict mapping each PROPHET_COL to its Pearson r.
    """
    # Generate Prophet forecasts
    forecast_dates = _get_datetime_series(fold_df)
    prophet_df = get_prophet_forecast(fold_df.copy(), forecast_dates, CONFIG)
    # Merge back
    merged = pd.concat([fold_df, prophet_df], axis=1)
    results: dict[str, float] = {}
    for col in PROPHET_COLS:
        if col not in merged.columns:
            results[col] = float('nan')
            continue
        x = pd.to_numeric(merged[col], errors='coerce').values
        y = pd.to_numeric(merged[TARGET], errors='coerce').values
        r = safe_pearson(x, y)
        results[col] = r if r is not None else 0.0
        print(f"[Diagnostic] Prophet-only corr {col}: {results[col]:.4f}")
    return results


def build_correlation_table(
    folds: List[pd.DataFrame], *, n_climate: int, n_econ: int
):
    per_fold: List[dict[str, float]] = []
    # Ensure Prophet cols appear in table even if correlation is zero
    union_feats: set[str] = set(PROPHET_COLS)
    for i, fld in enumerate(folds, start=1):
        print(f"\n[Diagnostic] Building correlations for Fold-{i}")
        d = dict(correlations_for_fold(fld, n_climate, n_econ))
        per_fold.append(d)
        union_feats.update(d)
    if not union_feats:
        raise RuntimeError("No correlations computed – check selector or data.")
    wide = pd.DataFrame(index=sorted(union_feats))
    for idx, d in enumerate(per_fold, start=1):
        wide[f"Fold-{idx}"] = pd.Series(d)
    wide.fillna(0.0, inplace=True)
    print(f"[Diagnostic] final correlation table index: {list(wide.index)}")
    return wide, per_fold

# ─── plotting ─────────────────────────────────────────────────────────────

def plot_heatmaps(per_fold: List[dict[str, float]], scheme: str) -> None:
    for idx, d in enumerate(per_fold, start=1):
        if not d:
            print(f"[Diagnostic] no features to plot for Fold-{idx}")
            continue
        df = pd.DataFrame({f"Fold-{idx}": d})
        # Order features alphabetically
        df = df.reindex(sorted(df.index), axis=0)
        print(f"[Diagnostic] Heatmap data for Fold-{idx}:\n{df}")
        longest = max(len(s) for s in df.index)
        width   = max(5.0, 2.0 + longest * 0.12)
        height  = max(3.0,       len(df)  * 0.40)
        fig, ax = plt.subplots(figsize=(width, height))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="vlag", center=0,
                     cbar_kws={"label": "Pearson r"}, ax=ax, annot_kws={"size": 8})
        ax.set_ylabel("")
        ax.set_xlabel("")
        plt.title(f"Feature–Target Correlation | {scheme} | Fold {idx}")
        fig.tight_layout()
        out_path = Path(f"heatmap_{scheme}_Fold-{idx}.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"✓ heat-map saved to {out_path}")

# ─── scheme runner ────────────────────────────────────────────────────────

def run_for_scheme(name: str, cfg: dict):
    print(f"\n=== Processing scheme: {name} ===")

    # Load & preprocess once
    base_df = preprocess_data(load_and_combine_duplicates().copy(), CONFIG)
    folds = split_into_folds(base_df, **cfg)
    print(f"✓ created {len(folds)} folds")

    enriched_folds: List[pd.DataFrame] = []
    for idx, fld in enumerate(folds, start=1):
        print(f"[Diagnostic] Generating Prophet forecast for Fold-{idx}")
        # Diagnostic: mirror regressor selection from ProphetQLSTMV3
        try:
            train_for_diag = fld.copy().rename(columns={'date':'ds','LoadConsumption':'y'})
            # add lag/molative features
            train_for_diag['lag_1'] = train_for_diag['y'].shift(1).ffill()
            train_for_diag['rolling_24'] = train_for_diag['y'].rolling(window=24, min_periods=1).mean().ffill()
            corr_diag = train_for_diag.corr(numeric_only=True)['y'].abs().sort_values(ascending=False)
            n_feat = CONFIG.get('n_features_to_select', 0)
            top_feats = [c for c in corr_diag.index if c != 'y'][:n_feat] + ['lag_1', 'rolling_24']
            print(f"[Diagnostic] regressors selected: {top_feats}")
        except Exception as ee:
            print(f"[Diagnostic] failed to select regressors: {ee}")

        fld_enriched = fld
        try:
            forecast_dates = _get_datetime_series(fld)
            # Use wrapper to respect advanced/basic logic and correct regressor selection
            prophet_df = get_prophet_forecast(fld.copy(), forecast_dates, CONFIG)
            print(f"[Diagnostic] prophet_df columns: {list(prophet_df.columns)}; head:\n{prophet_df.head(3)}")
            # Compute separate Prophet-only correlations on original fold
            print(f"[Diagnostic] Prophet-only correlations (pre-merge) for Fold-{idx}:")
            compute_prophet_only_correlation(fld)
            fld_enriched = pd.concat([fld, prophet_df], axis=1)
            # Compute separate Prophet-only correlations
            print(f"[Diagnostic] Prophet-only correlations for Fold-{idx}:")
            compute_prophet_only_correlation(fld_enriched)
            # Store Prophet-only dataframe for inspection
            merged = pd.concat([fld, prophet_df], axis=1)
            prophet_only_df = merged[[TARGET] + PROPHET_COLS]
            # Order columns alphabetically
            prophet_only_df = prophet_only_df.reindex(sorted(prophet_only_df.columns), axis=1)
            csv_prophet_only = Path(f"prophet_only_{name}_Fold-{idx}.csv")
            prophet_only_df.to_csv(csv_prophet_only)
            print(f"✓ saved Prophet-only data to {csv_prophet_only}")
            print(f"  • Prophet forecast merged into Fold-{idx} (cols now {fld_enriched.shape[1]})")
        except Exception as e:
            print(f"  ! Prophet forecast failed in Fold-{idx} → raw data used ({e})")
        enriched_folds.append(fld_enriched)

    wide, per_fold = build_correlation_table(
        enriched_folds,
        n_climate=CONFIG.get("n_climate_features", 7),
        n_econ=CONFIG.get("n_econ_features", 10),
    )

    csv_path = Path(f"fold_feature_correlations_{name}.csv")
    wide.to_csv(csv_path)
    print(f"✓ correlation table saved to {csv_path}")

    plot_heatmaps(per_fold, name)

# ─── entrypoint ───────────────────────────────────────────────────────────

def main():
    for scheme, cfg in SCHEMES.items():
        run_for_scheme(scheme, cfg)
    print("\nAll fold schemes processed.")


if __name__ == "__main__":
    main()
