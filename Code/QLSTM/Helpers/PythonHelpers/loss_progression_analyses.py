#!/usr/bin/env python3
"""
extract_and_plot_true_losses.py

This script can run in two modes:

1) **Extraction + Plotting** (default mode when given 2 args):
   - Walk through all `err_*.err` files in a folder.
   - Parse out only the true `loss=...` lines (classical until epoch resets, then quantum).
   - Dump a combined CSV.
   - Produce per-fold plots of end-of-epoch losses for classical vs quantum.

   Usage:
       python extract_and_plot_true_losses.py <err_dir> <output_csv>

2) **Plotting Only** (when given 1 arg):
   - Load an existing consolidated CSV of true losses.
   - Plot per-fold end-of-epoch loss curves.

   Usage:
       python extract_and_plot_true_losses.py <input_csv>
"""
import sys, os, re, glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_true_loss_file(path):
    fn = os.path.basename(path)
    base = fn[:-4]
    jobname, jobid, fold_str = base.split("_")
    fold = int(fold_str)
    records = []

    text = open(path, 'rb').read().replace(b'\r', b'\n').decode('utf-8', 'ignore')
    lines = text.split('\n')

    prev_ep = None
    variant = 'classical'

    for L in lines:
        # detect epoch resets
        m_ep = re.search(r"Epoch\s+(\d+)", L)
        if m_ep:
            ep = int(m_ep.group(1))
            if prev_ep is not None and ep < prev_ep:
                variant = 'quantum'
            prev_ep = ep

        # capture true loss= lines
        m_loss = re.search(r"([0-9]+)\/([0-9]+).*loss=([0-9]+\.[0-9]+)", L)
        if m_loss and prev_ep is not None:
            batch = int(m_loss.group(1))
            total = int(m_loss.group(2))
            loss = float(m_loss.group(3))
            records.append({
                'jobname': jobname,
                'jobid': jobid,
                'fold': fold,
                'variant': variant,
                'epoch': prev_ep,
                'batch': batch,
                'total_batches': total,
                'loss': loss,
                'filename': fn
            })
    return records


def extract_and_save(err_dir, out_csv):
    all_records = []
    for path in sorted(glob.glob(os.path.join(err_dir, 'err_*.err'))):
        all_records.extend(parse_true_loss_file(path))

    if not all_records:
        print("No true loss records found. Exiting.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    df.to_csv(out_csv, index=False)
    print(f"Saved consolidated loss CSV: {out_csv} (rows={len(df)})")
    return out_csv


def plot_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # sanity check
    for col in ['fold','variant','epoch','batch','loss']:
        if col not in df.columns:
            print(f"ERROR: Missing '{col}' column in CSV.")
            sys.exit(1)
    # Coerce types
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df['batch'] = pd.to_numeric(df['batch'], errors='coerce')
    df['loss']  = pd.to_numeric(df['loss'],  errors='coerce')

    out_dir = os.path.dirname(csv_path) or '.'
    for fold in sorted(df['fold'].unique()):
        sub = df[df['fold'] == fold]
        # For each variant pick the last batch per epoch
        plots = {}
        for variant in ['classical','quantum']:
            vdf = sub[sub['variant']==variant]
            if vdf.empty:
                plots[variant] = pd.DataFrame()
                continue
            # Group by epoch and pick row with max batch
            idx = vdf.groupby('epoch')['batch'].idxmax()
            plots[variant] = vdf.loc[idx].sort_values('epoch')

        cls_end = plots['classical']
        qtv_end = plots['quantum']

        plt.figure()
        if not cls_end.empty:
            plt.plot(cls_end['epoch'], cls_end['loss'], 'o-', label='Classical')
        if not qtv_end.empty:
            plt.plot(qtv_end['epoch'], qtv_end['loss'], 'x--', label='Quantum')
        plt.title(f'Fold {fold} End-of-Epoch Loss Progression')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_png = os.path.join(out_dir, f'loss_progression_fold_{fold}.png')
        plt.savefig(out_png)
        plt.close()
        print(f'Saved plot: {out_png}')

    # ------------------------------------------------------------------
    # ► NEW: create summary graphics in ./media/generated_loss_plots
    # ------------------------------------------------------------------
    SUM_DIR = os.path.join(out_dir, 'media', 'generated_loss_plots')
    os.makedirs(SUM_DIR, exist_ok=True)

    # 1) ── Representative fold (lowest quantum-validation loss) ─────────
    # pick fold with lowest FINAL quantum loss
    quantum_df = df[df['variant'] == 'quantum']
    if not quantum_df.empty:
        rep_fold = (
            quantum_df
            .groupby('fold')['loss']
            .last()          # last row per fold == last batch of last epoch
            .idxmin()
        )

        rep_sub = df[df['fold'] == rep_fold]
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for var, m, ls in [('classical', 'o', '-'), ('quantum', 'x', '--')]:
            vdf = rep_sub[rep_sub['variant'] == var]
            if vdf.empty:
                continue
            end_of_epoch = vdf.groupby('epoch')['batch'].idxmax()
            ax.plot(
                vdf.loc[end_of_epoch, 'epoch'],
                vdf.loc[end_of_epoch, 'loss'],
                marker=m, linestyle=ls, label=var.capitalize()
            )

        ax.set_title(f'Representative Fold {rep_fold}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation loss')
        ax.grid(True)
        ax.legend()
        rep_png = os.path.join(SUM_DIR, 'loss_rep_fold.png')
        fig.tight_layout()
        fig.savefig(rep_png, dpi=300)
        plt.close(fig)

        # 2) ── Mean & 95 % CI across 13 folds  ─────────────────────────────
        # first, align epoch axis (pad shorter runs with NaNs)
        pivot = (
            df.groupby(['variant', 'fold', 'epoch'])['loss']
            .last()      # last batch in epoch
            .unstack('epoch')
        )

        mean_loss = pivot.groupby(level='variant').mean(numeric_only=True)
        sem = pivot.groupby(level='variant').sem(numeric_only=True)
        ci95 = sem * 1.96

        fig, ax = plt.subplots(figsize=(4.5, 3))
        for var, ls in [('classical', '-'), ('quantum', '--')]:
            if var not in mean_loss.index:                 # safeguard
                continue
            epochs = mean_loss.columns.astype(int)
            ax.plot(epochs, mean_loss.loc[var], ls, label=var.capitalize())
            ax.fill_between(
                epochs,
                mean_loss.loc[var] - ci95.loc[var],
                mean_loss.loc[var] + ci95.loc[var],
                alpha=0.25
            )

        ax.set_title('Mean validation loss ±95 % CI')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True)
        ax.legend()
        mean_png = os.path.join(SUM_DIR, 'loss_mean_ci.png')
        fig.tight_layout()
        fig.savefig(mean_png, dpi=300)
        plt.close(fig)

        print(f'▲ Summary plots saved to: {SUM_DIR}')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        # Plotting only
        csv_path = sys.argv[1]
        plot_from_csv(csv_path)
    elif len(sys.argv) == 3:
        # Extraction + plotting
        err_dir, out_csv = sys.argv[1], sys.argv[2]
        csv_file = extract_and_save(err_dir, out_csv)
        plot_from_csv(csv_file)
    else:
        print(__doc__)
        sys.exit(1)



    # --- path to the consolidated loss CSV you just generated ---
    csv_path = Path("/Users/ruben/Documents/GitHub/MsCThesisRubenCuriel2024/Code/QLSTM/Results/LossPlotsAndSummary.csv")

    # 1) Load
    df = pd.read_csv(csv_path)

    # 2) Pick the *last batch* for every (variant, fold, epoch)
    final_rows = (df
        .groupby(['variant', 'fold', 'epoch'], as_index=False)
        .apply(lambda g: g.loc[g['batch'].idxmax()])
        .reset_index(drop=True))

    # 3) Aggregate across folds  →  mean, SEM, 95 % CI
    summary = (final_rows
        .groupby('variant', as_index=False)
        .agg(mean_loss=('loss', 'mean'),
             sem_loss =('loss', 'sem')))

    summary['ci95'] = 1.96 * summary['sem_loss']   # 95 % confidence interval

    # 4) Save + print
    out_csv = csv_path.parent / "final_loss_summary.csv"
    summary.to_csv(out_csv, index=False)
    stop_epochs = (df.groupby(['fold', 'variant'])['epoch']
                     .max()                       # last epoch actually logged
                     .unstack())                  # columns: classical | quantum

    # Save or print
    print("===== Last epoch per fold =====")
    print(stop_epochs)

    # Small summary
    summary_stop = stop_epochs.agg(['median', 'min', 'max'])
    print("\n===== Stop-epoch summary =====")
    print(summary_stop)

    print("\n===== End-of-epoch loss summary =====")
    print(summary.to_string(index=False))
    print(f"\n(saved to {out_csv})")
