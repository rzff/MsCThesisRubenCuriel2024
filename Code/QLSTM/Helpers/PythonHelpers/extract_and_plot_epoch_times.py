#!/usr/bin/env python3
"""
extract_and_plot_epoch_times.py
===============================
• Parses err_*.err log files produced during training.
• Generates a per-fold cumulative‑time plot **and** stores each fold's data
  in its own CSV.
• Builds a final bar chart showing total wall‑clock hours per fold
  (separately for Classical vs Quantum) by *summing the true per‑epoch
  runtimes* — no double counting.

**New in this drop‑in**
----------------------
* **Stricter deduplication** – for every *(fold, variant, epoch)* **only one
  row** is kept, even if multiple jobs/log files recorded the same epoch.
  (Fixes the duplicate‑epoch issue you saw in Fold 10.)

CLI
----
# Parse logs ➜ main CSV ➜ all plots
python extract_and_plot_epoch_times.py <err_dir> <output_csv>

# Re‑plot later from the main CSV
python extract_and_plot_epoch_times.py <output_csv>
"""

import sys
import os
import re
import glob
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────── Regex & Utils ────────────────────────────
_TIME_RE = re.compile(r"""
    Epoch\s+(\d+)\s+Batches:.*?  # epoch header
    100%.*?                       # tqdm finished
    \[                            # "["
        (\d+:\d+(?::\d+)?)       # mm:ss  or  hh:mm:ss
        <                         # separates elapsed & ETA
""", re.VERBOSE)

def _sec(hhmmss: str) -> float:
    """Convert ``hh:mm:ss`` or ``mm:ss`` string to *seconds*."""
    parts = list(map(int, hhmmss.split(":")))
    return parts[-1] + 60 * parts[-2] + (3600 * parts[0] if len(parts) == 3 else 0)

# ───────────────────────────── Parsing ────────────────────────────────

def parse_file(path: str) -> List[dict]:
    """Return one dict per *epoch* 100 % line in the given ``err_*.err`` file."""
    fn = os.path.basename(path)
    jobname, jobid, fold_str = fn[:-4].split("_")
    fold = int(fold_str)

    variant = "classical"
    prev_ep: Optional[int] = None
    recs: List[dict] = []

    with open(path, "rb") as f:
        text = f.read().replace(b"\r", b"\n").decode("utf‑8", "ignore")

    for line in text.split("\n"):
        # Detect variant switch
        m_ep = re.search(r"Epoch\s+(\d+)", line)
        if m_ep:
            ep = int(m_ep.group(1))
            if prev_ep is not None and ep < prev_ep:
                variant = "quantum"
            prev_ep = ep

        # Capture 100 % line
        m = _TIME_RE.search(line)
        if m:
            ep = int(m.group(1))
            hours = _sec(m.group(2)) / 3600.0
            recs.append({
                "jobname": jobname,
                "jobid": jobid,
                "fold": fold,
                "variant": variant,
                "epoch": ep,
                "hours": hours,
                "filename": fn,
            })
    return recs

# ────────────────────── Extraction (logs ➜ main CSV) ──────────────────

def _per_epoch(series: pd.Series) -> pd.Series:
    """Return *per‑epoch* runtime for one (fold, variant).

    Heuristic:
    ▸ If the elapsed timer is **clearly cumulative** (monotonic‑increasing **and**
      the final reading is much larger than the typical early reading) then we
      take a first‑difference to get per‑epoch hours.
    ▸ Otherwise we assume each value is already the runtime of that epoch.
    """
    is_cumulative = (
        series.is_monotonic_increasing
        and series.iloc[-1] > series.head(3).mean() * 3   # ≫ early epochs
    )
    if is_cumulative:
        return series.diff().fillna(series)
    return series
    return series.diff().fillna(series)

def extract(err_dir: str, out_csv: str) -> str:
    rows: List[dict] = []
    for f in sorted(glob.glob(os.path.join(err_dir, "err_*.err"))):
        rows.extend(parse_file(f))
    if not rows:
        sys.exit("No 100 % lines found – check regex or logs.")

    df = pd.DataFrame(rows)

    # ── NEW: strict deduplication ─────────────────────────────
    # Keep exactly *one* row per (fold, variant, epoch): the one with the
    # **largest** `hours` value (latest timestamp) across any jobs/logs.
    df = (
        df.sort_values(["fold", "variant", "epoch", "hours"])
          .groupby(["fold", "variant", "epoch"], as_index=False)
          .tail(1)         # last row per group == largest hours (thanks to sort)
    )

    df.sort_values(["fold", "variant", "epoch"], inplace=True)

    # Per‑epoch and cumulative hours
    df["epoch_hours"] = (
        df.groupby(["fold", "variant"], as_index=False)["hours"].transform(_per_epoch)
    )
    df["cum_hours"] = (
        df.groupby(["fold", "variant"], as_index=False)["epoch_hours"].cumsum()
    )

    df.to_csv(out_csv, index=False)
    print(f"Clean CSV written → {out_csv}  (rows={len(df)})")
    return out_csv

# ───────────────────────── Plot helpers ───────────────────────────────

def _plot_fold(df: pd.DataFrame, fold: int, out_dir: str) -> str:
    """Plot cumulative curve for one fold and save its CSV; return CSV path."""
    sub = df[df.fold == fold]
    if sub.empty:
        return ""

    plt.figure()
    for v, m, ls in [("quantum", "x", "--")]:  # hide classical
        d = sub[sub.variant == v]
        if d.empty:
            continue
        plt.plot(d.epoch, d.cum_hours, m, linestyle=ls, label=v.capitalize())

    plt.title(f"Fold {fold} – Cumulative Runtime")
    plt.xlabel("Epoch"); plt.ylabel("Cumulative Hours")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(); plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"epoch_time_fold_{fold}.png")
    plt.savefig(plot_path); plt.close()

    csv_path = os.path.join(out_dir, f"fold_{fold}_data.csv")
    sub.to_csv(csv_path, index=False)
    print(f"Saved plot → {plot_path}\nSaved data → {csv_path}")
    return csv_path

def _summary_plot(fold_csvs: List[str], base_dir: str):
    """Build bar chart of Σ epoch_hours from per‑fold CSVs."""
    if not fold_csvs:
        print("No per‑fold CSVs found; summary plot skipped.")
        return

    parts = []
    for p in fold_csvs:
        df = pd.read_csv(p)
        tot = (
            df.groupby(["fold", "variant"], as_index=False)["epoch_hours"]
              .sum()
              .set_index(["fold", "variant"])["epoch_hours"]
        )
        parts.append(tot)

    cum_fold = pd.concat(parts).unstack(level="variant", fill_value=0.0).sort_index()

    folds = cum_fold.index.astype(int).tolist()
    xs, w = range(len(folds)), 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4))

    if "quantum" in cum_fold:
        ax.bar([i + w/2 for i in xs], cum_fold["quantum"], w, label="Quantum", color="orange")

    ax.set_xticks(xs); ax.set_xticklabels(folds)
    ax.set_xlabel("Fold"); ax.set_ylabel("Total Quantum Hours (Σ epoch)")
    ax.set_title("Total Training Time per Fold")
    ax.grid(axis="y", linestyle="--", alpha=0.6); ax.legend(); fig.tight_layout()

    summary_dir = os.path.join(base_dir, "summary_plots")
    os.makedirs(summary_dir, exist_ok=True)
    fig.savefig(os.path.join(summary_dir, "total_time_per_fold.png"), dpi=300)
    plt.close()
    cum_fold.to_csv(os.path.join(summary_dir, "total_time_per_fold.csv"))
    print(f"Summary plot & CSV saved in {summary_dir}")

# ─────────────────────────── Orchestrator ─────────────────────────────

def orchestrate(main_csv: str):
    df = pd.read_csv(main_csv)
    for col in ["fold", "variant", "epoch", "epoch_hours", "cum_hours"]:
        if col not in df.columns:
            sys.exit(f"CSV missing '{col}'; regenerate with two‑arg mode.")

    df.sort_values(["fold", "variant", "epoch"], inplace=True)

    base_dir = os.path.dirname(main_csv) or "."
    per_fold_dir = os.path.join(base_dir, "per_fold_data")
    os.makedirs(per_fold_dir, exist_ok=True)

    fold_csvs = [
        p for f in sorted(df.fold.unique())
        if (p := _plot_fold(df, f, per_fold_dir))
    ]

    _summary_plot(fold_csvs, base_dir)

# ───────────────────────────── CLI ────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 2:           # plot‑only mode
        orchestrate(sys.argv[1])
    elif len(sys.argv) == 3:         # extract + plot mode
        orchestrate(extract(sys.argv[1], sys.argv[2]))
    else:
        print(__doc__)
        sys.exit(1)
