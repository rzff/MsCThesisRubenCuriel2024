#!/usr/bin/env python
"""
extract_epoch_times.py  –  turn a multi-fold training log into two CSV files:

  • epoch_times.csv   – fold | model | epoch | time_str | time_sec | loss
  • model_metrics.csv – fold | model | rmse  | mape     | pcc
    (one row each for Classical LSTM, QLSTM, Stacked Model)
"""

import re
import pathlib as pl
import pandas as pd

# ───── configuration ─────
LOG_FILE   = "/Users/ruben/Downloads/logs/TrainingLog4.rtf"      # adjust if needed
EPOCH_CSV  = "epoch_times.csv"
METRIC_CSV = "model_metrics.csv"

# ───── helper functions ─────
def time_to_sec(ts: str) -> float:           # 'MM:SS' or 'H:MM:SS' → seconds
    parts = list(map(float, ts.split(":")))
    if len(parts) == 2: parts = [0] + parts
    h, m, s = parts
    return h*3600 + m*60 + s

def num(x: str) -> float:                    # '1,234.56' → 1234.56
    return float(x.replace(",", ""))

# ───── regex patterns ─────
fold_re  = re.compile(r">>>\s*Running\s+fold\s+(\d+)", re.I)
model_re = re.compile(r">>>\s*Training\s+(Classical LSTM|QLSTM)", re.I)

epoch_re = re.compile(
    r"""Epoch\s+(\d+)          # epoch number
        .*?Batches:.*?\[
        (?P<t>[0-9:]+)         # wall-clock runtime
        .*?loss=\s*([0-9.]+)   # loss value
    """, re.VERBOSE)

# accepts Classical LSTM, QLSTM, Quantum LSTM, or Stacked Model (any prefix chars)
results_re = re.compile(
    r""".*?
        (Classical\s+LSTM|Quantum\s+LSTM|QLSTM|Stacked\s+Model)   # model name
        .*?RMSE:\s*([0-9.,]+)
        .*?MAPE:\s*([0-9.,]+)\s*%
        .*?PCC:\s*([-0-9.]+)
    """, re.I | re.VERBOSE)

# ───── main pass ─────
lines = pl.Path(LOG_FILE).read_text(encoding="utf-8", errors="ignore").splitlines()

epoch_rows, metric_rows = [], []
seen_summary            = set()          # (fold, model) already written
fold, model = None, None

for ln in lines:

    if (m := fold_re.match(ln)):                 # new fold
        fold, model = int(m.group(1)), None
        continue

    if (m := model_re.match(ln)):                # new model block
        model = m.group(1)                       # "Classical LSTM" | "QLSTM"
        continue

    if (m := epoch_re.match(ln)) and fold and model:
        epoch_rows.append({
            "fold":     fold,
            "model":    model,
            "epoch":    int(m.group(1)),
            "time_str": m.group("t"),
            "time_sec": time_to_sec(m.group("t")),
            "loss":     float(m.group(3)),       # ← fixed index (was 2)
        })
        continue

    if (m := results_re.search(ln)) and fold:
        raw_model, rmse, mape, pcc = m.groups()

        # normalise model name
        raw_upper = raw_model.upper()
        if "STACKED" in raw_upper:
            model_clean = "Stacked Model"
        elif "QUANTUM" in raw_upper or "QLSTM" in raw_upper:
            model_clean = "QLSTM"
        else:
            model_clean = "Classical LSTM"

        key = (fold, model_clean)
        if key in seen_summary:                  # skip duplicates
            continue
        seen_summary.add(key)

        metric_rows.append({
            "fold":  fold,
            "model": model_clean,
            "rmse":  num(rmse),
            "mape":  num(mape),
            "pcc":   float(pcc),
        })

# ───── write CSV files ─────
pd.DataFrame(epoch_rows).sort_values(["fold", "model", "epoch"]) \
  .to_csv(EPOCH_CSV, index=False)

metrics_df = pd.DataFrame(metric_rows)
if not metrics_df.empty:
    metrics_df.sort_values(["fold", "model"]).to_csv(METRIC_CSV, index=False)
    print(f"✓ wrote {len(epoch_rows):,} epoch rows  →  {EPOCH_CSV}")
    print(f"✓ wrote {len(metrics_df):,} metric rows →  {METRIC_CSV}")
else:
    print(f"✓ wrote {len(epoch_rows):,} epoch rows  →  {EPOCH_CSV}")
    print("⚠  No summary lines matched; model_metrics.csv not written")
