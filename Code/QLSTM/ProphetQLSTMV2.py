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
        m_ep = re.search(r"Epoch\s+(\d+)", L)
        if m_ep:
            ep = int(m_ep.group(1))
            if prev_ep is not None and ep < prev_ep:
                variant = 'quantum'
            prev_ep = ep

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
    for col in ['fold','variant','epoch','batch','total_batches','loss']:
        if col not in df.columns:
            print(f"ERROR: Missing '{col}' column in CSV.")
            sys.exit(1)
    # Coerce types
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df['loss']  = pd.to_numeric(df['loss'],  errors='coerce')
    df['batch'] = pd.to_numeric(df['batch'], errors='coerce')
    df['total_batches'] = pd.to_numeric(df['total_batches'], errors='coerce')

    out_dir = os.path.dirname(csv_path) or '.'
    for fold in sorted(df['fold'].unique()):
        sub = df[df['fold'] == fold]
        cls_end = sub[(sub['variant']=='classical') & (sub['batch']==sub['total_batches'])]
        qtv_end = sub[(sub['variant']=='quantum')  & (sub['batch']==sub['total_batches'])]

        plt.figure()
        plt.plot(cls_end['epoch'], cls_end['loss'], 'o-', label='Classical')
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
