#!/bin/bash

# Activate your virtual environment
source ~/venvs/qlstm_v2_venv/bin/activate

# Navigate to your project directory
cd ~/qlstm_prophetv2

# Run ProphetQLSTMV3.py for folds 1 through 8 sequentially
for FOLD in {12,10,11,1,2,3}; do
    echo ">>> Running fold $FOLD..."
    python ProphetQLSTMV4.py --fold $FOLD
done
