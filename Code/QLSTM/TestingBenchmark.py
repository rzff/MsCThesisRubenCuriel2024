import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from ProphetQLSTMV3 import (
    load_and_combine_duplicates,
    preprocess_data,
    create_sequences,
    QuantumLSTMModel,
    CONFIG,
    select_mixed_features_by_corr
)
from sklearn.preprocessing import MinMaxScaler
import time

if __name__ == "__main__":
    # Benchmark-specific config overrides
    benchmark_config = CONFIG.copy()
    benchmark_config.update({
        'seq_len': 12,
        'batch_size': 4,
        'n_qubits': 2,
        'n_qlayers': 1,
        'hidden_size1': 16,
        'hidden_size2': 32,
        'target_shift': 24 * 2,  # 2 days
        'epochs': 1,
        'use_dropout': False,
        'run_classical_lstm': False,
        'run_quantum_lstm': True,
    })

    df = load_and_combine_duplicates()
    df = df[(df['date'] >= '2010-01-01') & (df['date'] <= '2010-02-01')].copy()
    df = preprocess_data(df, benchmark_config)
    df = df.dropna(subset=['LoadConsumption'])

    selected = select_mixed_features_by_corr(
        df, target_col='LoadConsumption',
        n_climate=2, n_econ=2
    )

    scaler = MinMaxScaler().fit(df[selected])
    df[selected] = scaler.transform(df[selected])
    target_scaler = MinMaxScaler().fit(df[['LoadConsumption']])
    df['LoadConsumption'] = target_scaler.transform(df[['LoadConsumption']])

    X, y = create_sequences(df, selected, benchmark_config)
    if len(X) == 0:
        raise RuntimeError("Not enough data to create sequences.")

    loader = DataLoader(TensorDataset(X, y), batch_size=benchmark_config['batch_size'])

    model = QuantumLSTMModel(input_size=len(selected), config=benchmark_config)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Running benchmark on {len(X)} sequences with batch_size={benchmark_config['batch_size']}")
    print("Measuring forward pass time for 1 epoch...")

    start = time.time()
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            _ = model(batch_X)
    end = time.time()

    print(f"Total forward pass time: {end - start:.2f} seconds")
    print(f"Avg time per batch: {(end - start) / len(loader):.2f} seconds")
