import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

from ProphetQLSTMV3 import (
    load_and_combine_duplicates,
    preprocess_data,
    generate_prophet_forecast,
    generate_prophet_forecast_with_regressors,
    CONFIG
)

def plot_and_evaluate(y_true, yhat_base, yhat_adv, title_prefix=""):
    mape_base = mean_absolute_percentage_error(y_true, yhat_base) * 100
    mape_adv = mean_absolute_percentage_error(y_true, yhat_adv) * 100

    print(f"{title_prefix} Prophet MAPE (baseline): {mape_base:.2f}%")
    print(f"{title_prefix} Prophet MAPE (with regressors): {mape_adv:.2f}%")

    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="True", linewidth=2)
    plt.plot(yhat_base, label=f"Prophet (MAPE: {mape_base:.2f}%)")
    plt.plot(yhat_adv, label=f"Prophet+Regressors (MAPE: {mape_adv:.2f}%)")
    plt.legend()
    plt.title(f"{title_prefix} Prophet Forecast Comparison")
    plt.xlabel("Time")
    plt.ylabel("LoadConsumption")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_and_combine_duplicates()
    df = preprocess_data(df.copy(), CONFIG)

    forecast_dates = df['date'].copy()
    y_true = df['LoadConsumption'].dropna().values

    base_forecast = generate_prophet_forecast(df, forecast_dates, CONFIG)
    adv_forecast = generate_prophet_forecast_with_regressors(df, forecast_dates, CONFIG)

    yhat_base = base_forecast['yhat'].values
    yhat_adv = adv_forecast['yhat'].values

    # Ensure all arrays are the same length
    min_len = min(len(y_true), len(yhat_base), len(yhat_adv))
    plot_and_evaluate(
        y_true[:min_len],
        yhat_base[:min_len],
        yhat_adv[:min_len],
        title_prefix="Full Series"
    )
