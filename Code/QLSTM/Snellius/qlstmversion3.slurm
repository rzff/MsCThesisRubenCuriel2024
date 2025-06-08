#!/bin/bash
#SBATCH --job-name=qlstm_prophet_v2
#SBATCH --partition=rome            # default CPU partition (128 cores, 224 GB RAM)
#SBATCH --time=119:59:59            # ≤ 5 days, fits partition TIMELIMIT
#SBATCH --ntasks=1                  # one Python process
#SBATCH --cpus-per-task=8           # 8 cores accelerate Prophet & DataLoader
#SBATCH --mem=32G                   # ≈4 GB per core
#SBATCH --array=1-12%4               # run folds 1‑4, max 4 at a time (total ≤32 cores)
#SBATCH --output=logs/out_%A_%a.log
#SBATCH --error=logs/err_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rzff.curiel@gmail.com

# ----------  environment ----------
module load 2023
module load Python/3.10.4-GCCcore-11.3.0

# activate your virtual‑env
source ~/venvs/qlstm_v2_venv/bin/activate

# make sure BLAS/Prophet/PyTorch see the cores we requested
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ----------  run fold ----------
cd ~/qlstm_prophetv2
python ProphetQLSTMV4.py --fold ${SLURM_ARRAY_TASK_ID}
