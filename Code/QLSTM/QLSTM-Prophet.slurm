#!/bin/bash
#SBATCH --job-name=tune_q_lstm
#SBATCH --output=logs/tune_shift_%A_%a.out
#SBATCH --error=logs/tune_shift_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3

# Activate environment (adjust if needed)
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
source ~/venvs/qlstm/bin/activate

# Shift mapping: array index → shift value
SHIFTS=(168 336 720 1440 4380)
TARGET_SHIFT=${SHIFTS[$SLURM_ARRAY_TASK_ID]}

echo "Running tuning for shift: ${TARGET_SHIFT}"

# Run script with shift
python tune_config_cpu.py --target_shift "${TARGET_SHIFT}"
