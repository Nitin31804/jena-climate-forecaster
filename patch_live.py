import ast
import os
import re

utils_file = 'forecast_api/utils.py'
with open(utils_file, 'r') as f:
    code = f.read()

live_func = '''
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
'''

if 'get_live_weather_jena' not in code:
    code = code.replace('import pandas as pd', 'import pandas as pd\n' + live_func)

sarima_patch = '''def forecast_sarima(model, horizon, use_live_data=False):
    import pandas as pd
    import numpy as np
    if use_live_data:
        live_df = get_live_weather_jena().set_index("Date Time")
        updated_res = model.apply(live_df["T (degC)"])
        fc = updated_res.get_forecast(steps=horizon)
    else:
        fc = model.get_forecast(steps=horizon)
    return fc.predicted_mean.values.tolist(), fc.conf_int().iloc[:, 0].values.tolist(), fc.conf_int().iloc[:, 1].values.tolist()
'''
code = re.sub(r'def forecast_sarima.*?return.*?\n', sarima_patch, code, flags=re.DOTALL)

tft_patch = '''def forecast_tft(model, dataset, horizon, use_live_data=False):
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
'''
code = re.sub(r'def forecast_tft.*?return.*?\n', tft_patch, code, flags=re.DOTALL)

with open(utils_file, 'w') as f:
    f.write(code)

main_file = 'forecast_api/main.py'
with open(main_file, 'r') as f:
    mcode = f.read()

if 'use_live_data: bool' not in mcode:
    mcode = mcode.replace('horizon: int = Query(24, ge=1, le=168)', 'horizon: int = Query(24, ge=1, le=168), use_live_data: bool = Query(False)')
    mcode = mcode.replace('fc, low, up = forecast_sarima(sarima, horizon)', 'fc, low, up = forecast_sarima(sarima, horizon, use_live_data)')
    mcode = mcode.replace('fc, low, up = forecast_tft(tft, dataset, horizon)', 'fc, low, up = forecast_tft(tft, dataset, horizon, use_live_data)')
    with open(main_file, 'w') as f:
        f.write(mcode)
print("Patched successfully")
