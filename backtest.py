import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
from pytorch_forecasting import TimeSeriesDataSet
from forecast_api.utils import load_tft_model

df = pd.read_csv('df_hourly.csv', parse_dates=['Date Time'])
df = df.reset_index(drop=True)
df['time_idx'] = df.index
df['series_id'] = '0'
TARGET = 'T (degC)'

windows = [60000, 65000]
horizon = 24
train_size = 168

print('Loading TFT...')
try:
    tft, dataset = load_tft_model()
    tft_available = True
except Exception as e:
    tft_available = False
    print('TFT not available:', e)

results = []

for w in windows:
    # SARIMA
    train_data = df.iloc[w - train_size : w].copy()
    test_data = df.iloc[w : w + horizon].copy()
    actuals = test_data[TARGET].values

    print(f'Fitting SARIMA for window {w}...')
    sarima = SARIMAX(train_data[TARGET], order=(1,1,1), seasonal_order=(1,0,1,24), enforce_stationarity=False, enforce_invertibility=False)
    res = sarima.fit(disp=False, maxiter=20)
    sarima_fc = res.get_forecast(steps=horizon).predicted_mean.values
    sarima_mape = np.mean(np.abs((actuals - sarima_fc) / actuals)) * 100

    # TFT
    tft_mape = 'N/A'
    if tft_available:
        print(f'Inference TFT for window {w}...')
        # We need max_encoder_length + max_prediction_length rows for the dataloader context
        # Since predict=True assumes the end of the df is the end of the encoder context, we pass the 336 rows ending at w.
        encoder_df = df.iloc[w - (168 + 168) : w][["time_idx", "series_id", TARGET]].copy()
        
        pred_ds = TimeSeriesDataSet.from_dataset(dataset, encoder_df, predict=True, stop_randomization=True)
        pred_loader = pred_ds.to_dataloader(train=False, batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            x, _ = next(iter(pred_loader))
            preds = tft(x).prediction[:, :, 3].squeeze().cpu().numpy()
            if len(preds.shape) == 0: preds = np.array([preds])
            preds = preds[:horizon]
            if len(preds) < horizon: preds = np.pad(preds, (0, horizon - len(preds)))
        tft_mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    
    results.append({'Window End': w, 'SARIMA MAPE (%)': sarima_mape, 'TFT MAPE (%)': tft_mape})

res_df = pd.DataFrame(results)
print(res_df)
os.makedirs('assets', exist_ok=True)
res_df.to_markdown('assets/backtest_results.md', index=False)


