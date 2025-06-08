#!/bin/bash

# Activate your virtual environment
source ~/venvs/qlstm_v2_venv/bin/activate

# Navigate to your project directory
cd ~/qlstm_prophetv2

# Run ProphetQLSTMV3.py for folds 1 through 8 sequentially
for FOLD in {1,2,3,4,5,6,7,8,9,10,11,12}; do
    echo ">>> Running fold $FOLD..."
    python ProphetQLSTMV4.py --fold $FOLD
done
