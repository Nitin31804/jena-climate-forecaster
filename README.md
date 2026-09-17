# Jena Climate Forecaster

A reproducible temperature forecasting application for Jena, Germany. Compare a seasonal statistical model (SARIMA) with a Temporal Fusion Transformer (TFT), inspect hourly forecasts and uncertainty, and switch between historical station data and recent Open-Meteo weather.

## Run locally

Use Python 3.12. Run these commands from the project root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python export_hourly.py
python train_sarima.py
python train_tft.py
python backtest.py
python -m uvicorn forecast_api.main:app --host 127.0.0.1 --port 8000
```

Open [the dashboard](http://127.0.0.1:8000), [API documentation](http://127.0.0.1:8000/docs), or [readiness](http://127.0.0.1:8000/ready). The API serves the frontend itself; no Node build or chart CDN is required.

Training defaults to 10 CPU epochs, up to 100 training and 30 validation batches per epoch, batch size 32, and four Torch threads. This is a bounded demonstration experiment, not a fully optimized model. To use every batch, pass `--train-batches 0 --val-batches 0`. Set `--accelerator gpu` when suitable hardware is available. A smoke run such as `python train_tft.py --epochs 1 --train-batches 2 --val-batches 2` checks the pipeline but produces a poorly trained model. Every run records its settings, and replaces the artifact only after successful training.

Dataset downloads and recent weather requests require network access. `python export_hourly.py --raw-csv /path/to/jena_climate_2009_2016.csv` can preprocess an existing raw file offline. Training and historical inference then work without network access. Models and data are generated locally and excluded from Git.

## Dashboard

- Select historical data or recent weather, SARIMA or TFT, and a 1–168-hour horizon.
- Inspect 48 hours of available history, forecast dates, point predictions, and uncertainty.
- See the data cutoff, source, time convention, model version, and historical/demo label.
- Use the accessible hourly table or download a CSV containing dates, model version, mode, and interval level.
- Loading, timeout, missing-model, and upstream-error states are shown explicitly.

Historical mode forecasts after the final dataset hour in January 2017. It does **not** forecast today's weather. Historical timestamps preserve the dataset's wall-clock labels; their UTC offset is unspecified. Live mode requests UTC timestamps and uses only hours through the present. Open-Meteo supplies model-derived weather, which differs from direct station observations; live accuracy has not been validated against current station measurements.

## Pipeline and project map

| File | Purpose |
|---|---|
| `export_hourly.py` | Download, clean sentinels, aggregate hourly observations, and add causal features |
| `forecast_api/config.py` | Shared paths, model definitions, feature lists, and split ratios |
| `forecast_api/data.py` | Validate hourly data, cache source data, build future decoder rows, fetch recent weather |
| `forecast_api/training.py` | Shared split logic, fitted datasets, and training provenance |
| `train_sarima.py` | Fit and save compact statistical model parameters |
| `train_tft.py` | Train TFT and save the best validation epoch with fitted preprocessing |
| `forecast_api/models.py` | Load versioned artifacts once per worker and run inference |
| `forecast_api/utils.py` | Forecast response assembly and bounded result caching |
| `forecast_api/main.py` | API, readiness, and frontend hosting |
| `backtest.py` | Held-out evaluation against persistence and daily seasonal baselines |
| `frontend/` | Responsive dashboard using HTML, CSS, JavaScript, and SVG |
| `tests/` | Data, split, API, serialization, and inference regression tests |
| `jena_climate_eda.ipynb` | Historical exploratory notebook; not the supported training pipeline |

The data flow is raw 10-minute measurements → hourly data → chronological splits → versioned model artifacts → forecast API → dashboard.

## Data and leakage controls

Hourly bins are right-closed and labeled by their end, so no observation in a bin is later than its timestamp. Missing values are forward-filled using only past observations. `temperature_observed` identifies filled temperatures; they may serve as past context, but TFT training/validation target windows and evaluated forecast windows exclude them. Long filled gaps still reduce context quality and are a limitation.

The split is chronological: 70% training, 15% validation, and 15% held-out test data. TFT normalizers and feature scalers fit on training data only. Validation receives past encoder context, with every target strictly after the training cutoff. The lowest validation loss selects the saved epoch. Batch limits mean validation loss may cover only part of the validation period; this is recorded in the artifact.

At inference, 168 new decoder rows are appended strictly after the final observation. Temperature and its 6-hour/24-hour rolling means are past-only features. Daily and yearly sine/cosine features are the future-known inputs. Future temperature placeholders satisfy the dataset schema but are excluded from decoder inputs. Predictions are never zero-padded.

## Models and artifacts

**SARIMA:** order (1,1,1), seasonal order (1,0,1,24). Parameters are fitted on the last 720 hours of the training split and saved as JSON, with a convergence check. Inference filters up to 720 available past hours with these frozen parameters. The interval is the model's nominal 95% interval. A large serialized state-space result is unnecessary.

**TFT:** hidden size 16, one attention head, 168 encoder hours, and up to 168 prediction hours. The checkpoint contains weights, model configuration, fitted dataset encoders/scalers, quantiles, a small training sample needed to reconstruct the architecture, training cutoff, data fingerprint, dependency versions, seed, and training options. The forecast is the median; its interval is the 5th–95th quantiles (90% nominal coverage). Quantiles are sorted at inference to correct crossing. These bands are not interchangeable with SARIMA's 95% interval and are not empirically calibrated guarantees.

Artifacts live in `artifacts/sarima.json` and `artifacts/tft.pt`. Legacy `tft_checkpoint.pth` and `forecast_api/models/sarima.pkl` are deliberately ignored because they lack reproducible provenance. They can be removed manually when no longer needed. Only load TFT artifacts created by a trusted local training run: fitted Python preprocessing objects require deserialization.

Data/model caches are bounded and keyed by file modification signatures. Forecast results are cached by model version, source, history content, and horizon. Models load lazily once per process and automatically reload when artifacts change. Replacing historical data requires retraining; fingerprint mismatches return 503 instead of silently combining incompatible assets.

## Evaluation

```bash
python backtest.py --windows 12 --horizon 24
# Evaluate only SARIMA if TFT has not been trained:
python backtest.py --models sarima
```

The evaluator samples disjoint windows across the held-out test period. Frozen models receive only observations before each forecast origin; later test observations become available as context at later origins. It verifies artifact fingerprints and cutoffs, and skips windows containing imputed targets. It does not refit either model on test targets.

Results include MAE and RMSE in degrees Celsius, plus per-window interval coverage for learned models. Persistence repeats the most recent temperature; the seasonal baseline repeats the last 24 hours. Celsius MAPE is omitted because zero and negative values make percentage errors misleading. Reports in `assets/backtest_results.md` and `.json` record actual runs, model versions, training settings, and sampled dates. Results do not imply that a larger or longer-trained TFT will necessarily beat SARIMA or a baseline.

## API

`GET /forecast?city=Jena&model=sarima&horizon=24&use_live_data=false`

The response includes the original `city`, `model`, `horizon`, `forecast`, `lower_ci`, `upper_ci`, and generation `timestamp` fields, plus:

- `forecast_timestamps`, `data_cutoff`, and `timezone`
- `mode`, `data_source`, `model_version`, and `trained_through`
- `interval_level` and `interval_method`
- `history_timestamps` and `history` (up to 48 available hours)

`/health` is process liveness. `/ready?model=sarima` or `/ready?model=tft` verifies that local data and the selected artifact are loadable and compatible; it does not call the live provider. Invalid query parameters return 422 (unsupported city: 400), unavailable assets return 503, and recent-weather provider failures return 502. Unexpected errors are logged server-side with a generic client message. Upstream calls have a 10-second timeout and a five-minute cache. The browser imposes a 120-second request deadline.

Environment settings:

| Variable | Default |
|---|---|
| `JENA_DATA_PATH` | Project-root `df_hourly.csv` |
| `JENA_ARTIFACT_DIR` | Project-root `artifacts/` |
| `JENA_CORS_ORIGINS` | Empty; same-origin frontend. Comma-separated explicit origins if required. |

Set environment variables before launching a Python process. Multiple Uvicorn workers each load their own models; a single worker is the default for this demonstration.

## Verification and containers

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
node --check frontend/app.js
```

Tests cover future timestamps, causal features, missing observations, held-out splits, API validation/errors, caching, checkpoint round trips, and real TFT decoder behavior. They use synthetic data and mocked network access, so CI does not download weather or depend on local trained artifacts.

After generating data and artifacts locally:

```bash
docker compose up --build
```

The container and CI use CPU-only PyTorch wheels to avoid unnecessary GPU packages. The container runs as a non-root user and mounts only data and artifacts read-only. Large data, models, virtual environments, and notebook outputs are excluded from the image. Port 8000 binds to localhost. GitHub Actions runs the tests and JavaScript syntax check before building the image. The included configuration is a local deployment; public hosting would also need an HTTPS reverse proxy and deployment-specific access/rate controls.

Notebook dependencies are optional: `python -m pip install -r requirements-notebook.txt`. The notebook preserves earlier experiments and outputs, which are not the source of current benchmark claims.

## Sources and license

- [Jena Climate raw dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)
- [PyTorch Forecasting dataset API](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html)
- [Open-Meteo API documentation](https://open-meteo.com/en/docs)

Code is licensed under the included MIT LICENSE. Dataset and upstream-service terms remain those of their respective providers.
