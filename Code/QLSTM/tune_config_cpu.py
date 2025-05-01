import os
import joblib
import csv
import argparse
from skopt import forest_minimize
from skopt.space import Real, Integer, Categorical
from ProphetQLSTMV3 import CONFIG as BASE_CONFIG, load_and_combine_duplicates, preprocess_data, run_single_fold

# Define search space
space = [
    Integer(32, 128, name='batch_size'),
    Integer(32, 96, name='hidden_size1'),
    Integer(64, 128, name='hidden_size2'),
    Integer(2, 4, name='n_qubits'),
    Real(0.1, 0.5, name='dropout_rate'),
    Categorical([True, False], name='use_dropout'),
    Real(1e-4, 2e-3, prior='log-uniform', name='learning_rate'),
    Categorical([True, False], name='use_multiplicative_seasonality'),
    Categorical([True, False], name='use_advanced_prophet'),
    Integer(24, 72, name='seq_len'),
    Integer(6, 15, name='n_features_to_select'),
    Integer(2, 5, name='n_climate_features'),
    Integer(3, 8, name='n_econ_features'),
]

# Trial logger
def log_trial_result(params_dict, score, filepath='tuning_log.csv'):
    params_dict = params_dict.copy()
    params_dict['score'] = score
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=params_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(params_dict)

# Objective function
def objective(**params):
    config = BASE_CONFIG.copy()
    config.update(params)
    config['n_qlayers'] = 1
    config['quantum_backend'] = "lightning.qubit"
    config['run_classical_lstm'] = True
    config['run_quantum_lstm'] = True
    config['start_fold'] = 1
    config['epochs'] = 50
    config['patience'] = 5

    if not config['use_dropout']:
        config['dropout_rate'] = 0.0

    print("Running with config:", config)
    try:
        df = load_and_combine_duplicates()
        df_shifted = preprocess_data(df.copy(), config)
        result = run_single_fold(df_shifted, df_shifted, config, fold_id=1)
        score = result['mape'] if result else 99999
        print(f"Trial finished | MAPE: {score:.2f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Trial failed: {e}")
        score = 99999

    log_trial_result(params, score)
    return score

# Main tuning loop
if __name__ == "__main__":
    from pathlib import Path
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_shift', type=int, default=None, help='Single shift override')
    args = parser.parse_args()

    fixed_target_shifts = [args.target_shift] if args.target_shift else [168, 336, 720, 4380]
    n_trials_per_run = 10

    for shift in fixed_target_shifts:
        print(f"=== Running tuning for target_shift = {shift} ===")
        checkpoint_path = Path(f"skopt_checkpoint_shift_{shift}.pkl")
        log_file = f"tuning_log_shift_{shift}.csv"

        def objective_wrapped(params):
            params = dict(zip([dim.name for dim in space], params))
            params["target_shift"] = shift
            return objective(**params)

        if checkpoint_path.exists():
            print("Resuming from checkpoint...")
            res_old = joblib.load(checkpoint_path)
            res = forest_minimize(
                objective_wrapped,
                space,
                x0=res_old.x_iters,
                y0=res_old.func_vals,
                n_calls=len(res_old.x_iters) + n_trials_per_run,
                random_state=42,
                verbose=True
            )
        else:
            print("Starting fresh tuning run...")
            res = forest_minimize(
                objective_wrapped,
                space,
                n_calls=n_trials_per_run,
                random_state=42,
                verbose=True
            )

        joblib.dump(res, checkpoint_path)
        print("Best score so far for shift =", shift, ":", res.fun)
        print("Best parameters:")
        best_config = {name: val for name, val in zip([dim.name for dim in space], res.x)}
        best_config['target_shift'] = shift
        print(json.dumps(best_config, indent=2))
        with open(f"best_config_shift_{shift}.json", "w") as f:
            json.dump(best_config, f, indent=2)
