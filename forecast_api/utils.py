# forecast_api/utils.py
"""Utility functions for the FastAPI weather-forecast service.
Provides:
  * load_sarima_model() - fits a SARIMA on the whole series (or loads a cached pickle).
  * load_tft_model() - loads the trained TFT checkpoint and returns model + dataset.
  * forecast_sarima(model, horizon) - returns point forecast, lower and upper 95% CI.
  * forecast_tft(model, dataset, horizon) - returns median forecast and 5/95 quantile bounds.
"""

import os
import pandas as pd

def get_live_weather_jena():
    import urllib.request
    import json
    import pandas as pd
    url = "https://api.open-meteo.com/v1/forecast?latitude=50.9272&longitude=11.5861&past_days=14&hourly=temperature_2m"
    req = urllib.request.urlopen(url)
    data = json.loads(req.read())
    times = pd.to_datetime(data["hourly"]["time"])
    temps = data["hourly"]["temperature_2m"]
    df = pd.DataFrame({"Date Time": times, "T (degC)": temps})
    now = pd.Timestamp.utcnow().tz_localize(None)
    return df[df["Date Time"] <= now].dropna()

import numpy as np
import pickle
import torch

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SARIMA_PICKLE = os.path.join(BASE_DIR, "forecast_api", "models", "sarima.pkl")
DATA_CSV = os.path.join(BASE_DIR, "df_hourly.csv")
TFT_CHECKPOINT = os.path.join(BASE_DIR, "tft_checkpoint.pth")
TARGET = "T (degC)"


# ---------- SARIMA utilities ----------

def load_sarima_model():
    """Load a SARIMA model from pickle if it exists, otherwise fit on the full series.
    The model is saved to forecast_api/models/sarima.pkl for reuse.
    """
    if os.path.exists(SARIMA_PICKLE):
        with open(SARIMA_PICKLE, "rb") as f:
            model = pickle.load(f)
        return model
    # Fit on the entire series
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError("{} not found - run export_hourly.py first.".format(DATA_CSV))
    df = pd.read_csv(DATA_CSV, parse_dates=["Date Time"], index_col="Date Time")
    series = df[TARGET]
    model = SARIMAX(series, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=100)
    # Cache the fitted model
    os.makedirs(os.path.dirname(SARIMA_PICKLE), exist_ok=True)
    with open(SARIMA_PICKLE, "wb") as f:
        pickle.dump(results, f)
    return results


def forecast_sarima(model, horizon, use_live_data=False):
    import pandas as pd
    import numpy as np
    if use_live_data:
        live_df = get_live_weather_jena().set_index("Date Time")
        updated_res = model.apply(live_df["T (degC)"])
        fc = updated_res.get_forecast(steps=horizon)
    else:
        fc = model.get_forecast(steps=horizon)
    return fc.predicted_mean.values.tolist(), fc.conf_int().iloc[:, 0].values.tolist(), fc.conf_int().iloc[:, 1].values.tolist()


# ---------- TFT utilities ----------

def load_tft_model():
    """Load the trained Temporal Fusion Transformer and corresponding dataset.
    Returns (model, dataset) ready for inference.
    """
    if not os.path.exists(TFT_CHECKPOINT):
        raise FileNotFoundError("TFT checkpoint not found at {} - run train_tft.py first.".format(TFT_CHECKPOINT))

    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    df = pd.read_csv(DATA_CSV, parse_dates=["Date Time"])
    df = df.reset_index(drop=True)
    df["time_idx"] = df.index
    df["series_id"] = "0"
    df = df[["time_idx", "series_id", TARGET]].copy()

    # Build the same dataset definition used during training
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["series_id"],
        max_encoder_length=168,
        max_prediction_length=168,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[TARGET],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
    )

    # Initialise model with same hyper-parameters
    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        loss=QuantileLoss(),
    )
    # Load checkpoint weights
    tft.load_state_dict(torch.load(TFT_CHECKPOINT, map_location=torch.device("cpu")))
    tft.eval()
    return tft, dataset


def forecast_tft(model, dataset, horizon, use_live_data=False):
    import pandas as pd
    import numpy as np
    if use_live_data:
        df = get_live_weather_jena()
        df["time_idx"] = range(70106, 70106 + len(df))
    else:
        df = pd.read_csv(DATA_CSV, parse_dates=["Date Time"])
        df = df.reset_index(drop=True)
        df["time_idx"] = df.index
    df["series_id"] = "0"
    df = df[["time_idx", "series_id", TARGET]].copy()
    encoder_df = df.iloc[-(168 + 168):]
    from pytorch_forecasting import TimeSeriesDataSet
    pred_dataset = TimeSeriesDataSet.from_dataset(dataset, encoder_df, predict=True, stop_randomization=True)
    pred_loader = pred_dataset.to_dataloader(train=False, batch_size=1, shuffle=False, num_workers=0)
    import torch
    with torch.no_grad():
        x, _ = next(iter(pred_loader))
        output = model(x)
        preds = output.prediction
        median = preds[:, :, 3].squeeze().cpu().numpy()
        lower = preds[:, :, 0].squeeze().cpu().numpy()
        upper = preds[:, :, 6].squeeze().cpu().numpy()
    
    if len(median.shape) == 0:
        median = np.array([median])
        lower = np.array([lower])
        upper = np.array([upper])
        
    median = median[:horizon] if len(median) >= horizon else np.pad(median, (0, horizon - len(median)))
    lower = lower[:horizon] if len(lower) >= horizon else np.pad(lower, (0, horizon - len(lower)))
    upper = upper[:horizon] if len(upper) >= horizon else np.pad(upper, (0, horizon - len(upper)))
    return median.tolist(), lower.tolist(), upper.tolist()

    raise RuntimeError("Failed to generate TFT prediction.")
