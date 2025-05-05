#!/bin/bash
#SBATCH --job-name=qlstm_prophet_v2
#SBATCH --time=36:00:00                     # 36 hours per job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rzff.curiel@gmail.com
#SBATCH --array=1-8                         # Run 8 folds in parallel
#SBATCH --output=logs/output_fold_%A_%a.log
#SBATCH --error=logs/error_fold_%A_%a.err

# Load modules
module load 2023
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment
source ~/venvs/qlstm_v2_venv/bin/activate

# Navigate to your project directory
cd ~/qlstm_prophetv2

# Run the Python script for the current fold
python ProphetQLSTMV3.py --fold ${SLURM_ARRAY_TASK_ID}
