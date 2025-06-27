from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import json
from ProphetQLSTM import run_single_fold, CONFIG, load_and_combine_duplicates, preprocess_data

# --- Load and preprocess dataset ---
df = load_and_combine_duplicates()
df_shifted = preprocess_data(df.copy(), CONFIG)

# --- Train/test split ---
train_df = df_shifted[df_shifted['date'].dt.year < 2019].copy()
test_df = df_shifted[df_shifted['date'].dt.year == 2019].copy()

# --- Disable quantum model ---
CONFIG['run_classical_lstm'] = True
CONFIG['run_quantum_lstm'] = False

# --- Search space for Bayesian optimization ---
search_space = [
    Integer(2, 10, name='n_econ_features'),
    Integer(2, 10, name='n_climate_features')
]

results = []

@use_named_args(search_space)
def objective(**params):
    CONFIG['n_econ_features'] = params['n_econ_features']
    CONFIG['n_climate_features'] = params['n_climate_features']

    print(f"Running with N_Econ={params['n_econ_features']}, N_Climate={params['n_climate_features']}")

    result = run_single_fold(train_df, test_df, CONFIG, fold_id=f"{params['n_econ_features']}_{params['n_climate_features']}")
    if result is None:
        print("Run returned None (e.g. due to NaNs). Penalizing.")
        return 1e6  # Large RMSE penalty

    rmse = result.get("rmse", float("inf"))

    results.append({
        "N_Econ": int(params['n_econ_features']),
        "N_Climate": int(params['n_climate_features']),
        "RMSE": float(rmse),
        "MAPE": float(result.get("mape")),
        "PCC": float(result.get("pcc"))
    })

    return rmse

# --- Run Bayesian optimization ---
res = gp_minimize(objective, search_space, n_calls=25, random_state=42)

# --- Save results ---
with open("feature_selection_results.json", "w") as f:
    json.dump(results, f, indent=2)

# --- Output best combination ---
print("\nBest combination found:")
print(f"N_Econ: {res.x[0]}, N_Climate: {res.x[1]}")
print(f"Best RMSE: {res.fun:.2f}")
