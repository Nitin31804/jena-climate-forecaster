# Jena Climate Weather Forecasting

An end-to-end weather forecasting pipeline built on the **Jena Climate 2009-2016** dataset. The project demonstrates data preprocessing, time-series modeling with both classical (SARIMA) and deep learning (Temporal Fusion Transformer) approaches, walk-forward backtesting, and a REST API for serving predictions.

---

## Project Structure

```
.
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
│
├── export_hourly.py                 # Resample raw data to hourly + feature engineering
├── train_tft.py                     # Train TFT model (10 epochs (trained for 1 epoch on CPU for demo speed), CPU)
├── backtest.py                      # SARIMA and TFT walk-forward validation (2 windows)
├── jena_climate_eda.ipynb           # Jupyter notebook: EDA, visualizations, SARIMA baseline
│
├── tft_checkpoint.pth               # Saved TFT model weights
├── assets/
│   └── backtest_results.md          # Backtesting results table
│
└── forecast_api/                    # FastAPI inference service
    ├── __init__.py
    ├── main.py                      # API endpoints (/forecast, /health)
    ├── utils.py                     # Model loading & inference utilities
    ├── requirements.txt             # API-specific dependencies
    └── models/                      # Cached model files (auto-generated)
        └── sarima.pkl               # Fitted SARIMA model (created on first API call)
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data

Downloads the Jena Climate dataset, resamples to hourly, and engineers features:

```bash
python export_hourly.py
```

Output: `df_hourly.csv` (~70,000 rows, 21 columns)

### 3. Train the TFT Model

Trains a Temporal Fusion Transformer (15,909 parameters, 10 epochs (trained for 1 epoch on CPU for demo speed) on CPU):

```bash
python train_tft.py
```

Output: `tft_checkpoint.pth`

### 4. Run Backtesting

Evaluates SARIMA and TFT across 2 windows (168h training context, 24h forecast each):

```bash
python backtest.py
```

Output: `assets/backtest_results.md`

### 5. Start the API Server

```bash
python -m uvicorn forecast_api.main:app --host 127.0.0.1 --port 8000
```

Then visit:
- **Swagger docs**: http://127.0.0.1:8000/docs
- **Health check**: http://127.0.0.1:8000/health
- **SARIMA forecast**: http://127.0.0.1:8000/forecast?model=sarima&horizon=24
- **TFT forecast**: http://127.0.0.1:8000/forecast?model=tft&horizon=24

---

## API Reference

### `GET /forecast`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `city` | string | `Jena` | City name (only Jena supported) |
| `model` | string | `sarima` | Model: `sarima` or `tft` |
| `horizon` | int | `24` | Hours to forecast (1-168) |

**Response:**
```json
{
  "city": "Jena",
  "model": "sarima",
  "horizon": 24,
  "forecast": [5.12, 5.08, ...],
  "lower_ci": [3.01, 2.45, ...],
  "upper_ci": [7.23, 7.71, ...],
  "timestamp": "2026-08-20T12:45:00"
}
```

### `GET /health`

Returns `{"status": "ok"}` when the server is running.

---

## Models

### SARIMA (Baseline)
- **Order**: ARMA(1,0,1) -- no differencing, seasonal component (1,0,1,24)
- **Training**: Fits on the full hourly series
- **Inference**: ~1 second per forecast

### Temporal Fusion Transformer (TFT)
- **Architecture**: Attention-based encoder-decoder with variable selection
- **Parameters**: 15,909
- **Training**: 10 epochs (trained for 1 epoch on CPU for demo speed), 50 train batches, 10 val batches (CPU)
- **Library**: `pytorch-forecasting` 1.8.0

---

## Backtest Results
 
| Window End | SARIMA MAPE (%) | TFT MAPE (%) |
|------------|-----------------|--------------|
| 60000      | 9.79            | 33.71        |
| 65000      | 7.22            | 37.53        |

---

## Dataset

The [Jena Climate dataset](https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016) contains 14 weather features recorded every 10 minutes from 2009 to 2016 at the Max Planck Institute for Biogeochemistry in Jena, Germany.

| Feature | Unit |
|---------|------|
| T (degC) | Temperature |
| p (mbar) | Atmospheric pressure |
| rh (%) | Relative humidity |
| wv (m/s) | Wind velocity |
| wd (deg) | Wind direction |
| ... | + 9 more features |

---

## License

This project uses the Jena Climate dataset which is publicly available for research and educational purposes.
