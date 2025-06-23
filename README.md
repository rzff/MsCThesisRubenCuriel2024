# Enhancing energy demand forecasting using hybrid quantum computing with climate and economic predictors

This repository support a master thesis project of the University of Amsterdam (UvA) and focuses on enhancing energy demand forecasting using hybrid quantum computing with climate and economic predictors

## Overview

In this repository you find a forecasting pipeline that integrates classical machine learning with quantum-inspired models to predict electricity demand in the Netherlands. The project explores the performance of:

- Classical LSTM models  
- Quantum LSTM (QLSTM) architectures (based on PennyLane and PyTorch)  
- Prophet models (with exogenous regressors)  
- Hybrid ensembles using stacked generalization

All models are evaluated using rolling origin cross-validation (ROCV) and optimized through Bayesian hyperparameter tuning.

## Features

- Hybrid forecasting pipeline with Prophet and QLSTM
- Quantum LSTM integration via PennyLane and PyTorch
- Rolling Origin Cross-Validation for robust time series evaluation
- Feature selection via correlation 
- Stacked generalization combining Prophet with LSTM/QLSTM outputs
- Evaluation metrics: RMSE, MAPE, PCC
- Training checkpointing and early stopping
- Snellius HPC support for quantum simulation scalability

## Project Structure

- Code
  - Notebooks
  - QLSTM

 
## Requirements

- Python 3.10+
- PyTorch
- PennyLane
- Prophet
- NumPy, pandas, scikit-learn
- Matplotlib
- tqdm

Install dependencies via:

```bash
pip install -r requirements.txt

## Usage

### Run the main Prophet + QLSTM forecasting pipeline

```bash
python ProphetQLSTM.py
