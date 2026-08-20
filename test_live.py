import urllib.request
import json
import pandas as pd
import numpy as np

# 1. Fetch live data
url = "https://api.open-meteo.com/v1/forecast?latitude=50.9272&longitude=11.5861&past_days=14&hourly=temperature_2m"
req = urllib.request.urlopen(url)
data = json.loads(req.read())
times = pd.to_datetime(data["hourly"]["time"])
temps = data["hourly"]["temperature_2m"]
df = pd.DataFrame({"Date Time": times, "T (degC)": temps})
now = pd.Timestamp.utcnow().tz_localize(None)
df = df[df["Date Time"] <= now].dropna()

print("Fetched live data:", len(df), "rows. Last time:", df["Date Time"].iloc[-1])

# 2. Test SARIMA
from forecast_api.utils import load_sarima_model
print("Loading SARIMA...")
sarima = load_sarima_model()
print("Applying live data to SARIMA...")
# apply
live_series = df.set_index("Date Time")["T (degC)"]
updated_res = sarima.apply(live_series)
fc = updated_res.get_forecast(steps=3)
print("SARIMA forecast:", fc.predicted_mean.values)

# 3. Test TFT
from forecast_api.utils import load_tft_model
print("Loading TFT...")
tft, dataset = load_tft_model()
df["time_idx"] = range(70106, 70106 + len(df))
df["series_id"] = "0"
encoder_df = df.iloc[-336:]
from pytorch_forecasting import TimeSeriesDataSet
print("Creating pred dataset...")
pred_dataset = TimeSeriesDataSet.from_dataset(dataset, encoder_df, predict=True, stop_randomization=True)
pred_loader = pred_dataset.to_dataloader(train=False, batch_size=1, shuffle=False, num_workers=0)
import torch
with torch.no_grad():
    x, _ = next(iter(pred_loader))
    output = tft(x)
    preds = output.prediction
    median = preds[:, :, 3].squeeze().cpu().numpy()
    print("TFT forecast:", median[:3])
