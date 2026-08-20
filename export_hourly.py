# export_hourly.py
"""Extract the Jena Climate CSV, resample to hourly, engineer features, and save to CSV.
This script mirrors the preprocessing steps from the notebook so that downstream
scripts (TFT training, backtest) can load a ready‑to‑use file.
"""

import os
import pandas as pd
import numpy as np

# 1. Download the dataset (same as notebook) – TensorFlow utility
import urllib.request
import zipfile
# Download the dataset zip file
zip_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
zip_path = os.path.join(os.getcwd(), 'jena_climate_2009_2016.csv.zip')
if not os.path.exists(zip_path):
    urllib.request.urlretrieve(zip_url, zip_path)
# Extract the CSV from the zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.getcwd())
# The extracted CSV file name
csv_path = os.path.join(os.getcwd(), 'jena_climate_2009_2016.csv')

# 2. Load raw data
df = pd.read_csv(csv_path)

# 3. Parse datetime and set index
if 'Date Time' in df.columns:
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df.set_index('Date Time', inplace=True)

# 4. Resample to hourly (mean) and interpolate missing values
df_hourly = df.resample('H').mean()
df_hourly = df_hourly.interpolate(method='linear')

# 5. Feature engineering (same as notebook)
# Rolling temperature means
df_hourly['T_rolling_6h'] = df_hourly['T (degC)'].rolling(window=6).mean()
df_hourly['T_rolling_24h'] = df_hourly['T (degC)'].rolling(window=24).mean()
# Cyclical time features
day = 24 * 60 * 60
year = 365.2425 * day
timestamp_s = df_hourly.index.map(pd.Timestamp.timestamp)
df_hourly['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df_hourly['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df_hourly['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df_hourly['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
# Drop rows with NaNs from rolling windows
df_hourly.dropna(inplace=True)

# 6. Save to CSV in workspace root
out_path = 'df_hourly.csv'
df_hourly.to_csv(out_path)
print(f"Hourly pre-processed data saved to {out_path} with {len(df_hourly)} rows")
