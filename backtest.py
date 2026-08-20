# backtest.py
"""Walk-forward validation for SARIMA on hourly temperature data.
Picks 5 evenly-spaced evaluation windows across the dataset.
Each window uses 500 hours of training data and forecasts a 24-hour horizon.
Results are saved to assets/backtest_results.md as a Markdown table.
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ---------- Helper metric ----------
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ---------- Load data ----------
DATA_PATH = os.path.join(os.path.dirname(__file__), "df_hourly.csv")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Run preprocessing first.")

df = pd.read_csv(DATA_PATH, parse_dates=["Date Time"], index_col="Date Time")

# Target series
TARGET = "T (degC)"
series = df[TARGET].astype(float)

print("Loaded {} rows from df_hourly.csv".format(len(series)))

# ---------- SARIMA forecast function ----------
def forecast_sarima(train_series, steps):
    """Fit a simple ARMA(1,0,1) -- no differencing, no seasonal -- and forecast."""
    model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 24), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=20)
    forecast = results.get_forecast(steps=steps)
    return forecast.predicted_mean

# ---------- Pick 5 evenly-spaced evaluation windows ----------
window_size = 500   # training window length (hours)
horizon = 24        # forecast horizon (hours)
n_windows = 5

total_rows = len(series)
usable = total_rows - window_size - horizon  # max valid start index

if usable <= 0:
    raise ValueError("Dataset too small for window_size={} and horizon={}".format(window_size, horizon))

# Evenly-spaced start indices
starts = np.linspace(0, usable, n_windows, dtype=int)

print("Running {} evaluation windows (window_size={}, horizon={})".format(n_windows, window_size, horizon))
print("-" * 70)

results = []
for i, start in enumerate(starts):
    train_end = start + window_size
    train_series = series.iloc[start:train_end]
    true_future = series.iloc[train_end:train_end + horizon]

    train_start_dt = train_series.index[0]
    forecast_start_dt = true_future.index[0]

    print("Window {}/{}: train [{} ... {}], forecast starts {}".format(
        i + 1, n_windows, train_start_dt, train_series.index[-1], forecast_start_dt))

    # SARIMA forecast
    sarima_pred = forecast_sarima(train_series, horizon)

    sar_mae = mean_absolute_error(true_future, sarima_pred)
    sar_rmse = np.sqrt(mean_squared_error(true_future, sarima_pred))
    sar_mape = mape(true_future.values, sarima_pred.values)

    results.append({
        "Window": i + 1,
        "Train Start": str(train_start_dt),
        "Forecast Start": str(forecast_start_dt),
        "MAE (degC)": round(sar_mae, 4),
        "RMSE (degC)": round(sar_rmse, 4),
        "MAPE (%)": round(sar_mape, 4),
    })

    print("  => MAE={:.4f}, RMSE={:.4f}, MAPE={:.4f}%".format(sar_mae, sar_rmse, sar_mape))

# ---------- Summary ----------
df_results = pd.DataFrame(results)

avg_mae = df_results["MAE (degC)"].mean()
avg_rmse = df_results["RMSE (degC)"].mean()
avg_mape = df_results["MAPE (%)"].mean()

print("-" * 70)
print("AVERAGE across {} windows:  MAE={:.4f}  RMSE={:.4f}  MAPE={:.4f}%".format(
    n_windows, avg_mae, avg_rmse, avg_mape))

# ---------- Save markdown table ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(SCRIPT_DIR, "assets")
os.makedirs(assets_dir, exist_ok=True)
md_path = os.path.join(assets_dir, "backtest_results.md")

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Backtest Results -- SARIMA(1,0,1) Walk-Forward\n\n")
    f.write("- **Model**: ARMA(1,0,1) -- no differencing, no seasonal component\n")
    f.write("- **Training window**: {} hours\n".format(window_size))
    f.write("- **Forecast horizon**: {} hours\n".format(horizon))
    f.write("- **Number of windows**: {}\n\n".format(n_windows))
    f.write("## Per-Window Results\n\n")
    f.write(df_results.to_markdown(index=False))
    f.write("\n\n## Average Metrics\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write("| MAE (degC) | {:.4f} |\n".format(avg_mae))
    f.write("| RMSE (degC) | {:.4f} |\n".format(avg_rmse))
    f.write("| MAPE (%) | {:.4f} |\n".format(avg_mape))

print("\nResults saved to {}".format(md_path))
