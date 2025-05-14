#!/bin/bash

# Activate your virtual environment
source ~/venvs/qlstm_v2_venv/bin/activate

# Navigate to your project directory
cd ~/qlstm_prophetv2

# Run ProphetQLSTMV3.py for folds 1 through 8 sequentially
for FOLD in {1,3}; do
    echo ">>> Running fold $FOLD..."
    python ProphetQLSTMV3.py --fold $FOLD
done
