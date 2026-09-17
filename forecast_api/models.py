"""Versioned model artifacts and inference. Only load locally trusted TFT artifacts."""
import json
import os
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import FORMAT_VERSION, MAX_HORIZON, SARIMA_CONTEXT, SARIMA_PATH, SARIMA_SPEC, TARGET, TFT_PATH
from .data import DataUnavailable, prediction_frame, signature, validate_history


class ModelUnavailable(RuntimeError):
    """Model is missing, incompatible, or not ready for inference."""


MODEL_LOCK = RLock()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def validate_metadata(artifact: dict, model: str) -> dict:
    if not isinstance(artifact, dict):
        raise ModelUnavailable(f"Invalid {model.upper()} artifact. Retrain it.")
    meta = artifact.get("metadata", {})
    if not isinstance(meta, dict):
        raise ModelUnavailable(f"Invalid {model.upper()} metadata. Retrain it.")
    if artifact.get("format_version") != FORMAT_VERSION or meta.get("model") != model:
        raise ModelUnavailable(f"Incompatible {model.upper()} artifact. Retrain with the current scripts.")
    required = ["version", "data_fingerprint", "trained_through", "validation_through", "test_start", "training_rows"]
    if any(key not in meta for key in required):
        raise ModelUnavailable(f"{model.upper()} artifact lacks training provenance. Retrain it.")
    try:
        train, validation, test = [pd.Timestamp(meta[key]) for key in ["trained_through", "validation_through", "test_start"]]
        if not train < validation < test or int(meta["training_rows"]) < 1:
            raise ValueError("Invalid split chronology")
    except (ValueError, TypeError) as exc:
        raise ModelUnavailable(f"Invalid {model.upper()} training provenance. Retrain it.") from exc
    return meta


def fit_sarima(history, maxiter=100):
    values = validate_history(history, 48)[TARGET].iloc[-SARIMA_CONTEXT:].to_numpy()
    model = SARIMAX(values, **SARIMA_SPEC, enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False, maxiter=maxiter)


@lru_cache(maxsize=2)
def _load_sarima(key: tuple):
    artifact = json.loads(Path(key[0]).read_text(encoding="utf-8"))
    validate_metadata(artifact, "sarima")
    if artifact.get("spec") != {k: list(v) for k, v in SARIMA_SPEC.items()}:
        raise ModelUnavailable("SARIMA configuration changed. Retrain it.")
    if not artifact["metadata"].get("converged"):
        raise ModelUnavailable("SARIMA artifact did not converge. Retrain it with more iterations.")
    parameters = np.asarray(artifact["parameters"], dtype=float)
    if parameters.shape != (5,) or not np.isfinite(parameters).all() or parameters[-1] <= 0:
        raise ModelUnavailable("Invalid SARIMA parameters. Retrain it.")
    return artifact


def load_sarima(path: Path = SARIMA_PATH):
    try:
        with MODEL_LOCK:
            return _load_sarima(signature(path))
    except ModelUnavailable:
        raise
    except (DataUnavailable, ValueError, KeyError, TypeError, OSError) as exc:
        raise ModelUnavailable("SARIMA is unavailable. Run python train_sarima.py.") from exc


def predict_sarima(artifact, history, horizon):
    values = validate_history(history, 48)[TARGET].iloc[-SARIMA_CONTEXT:].to_numpy()
    model = SARIMAX(values, **SARIMA_SPEC, enforce_stationarity=False, enforce_invertibility=False)
    # Update state with available history and frozen parameters; never refit here.
    result = model.filter(np.asarray(artifact["parameters"]))
    fc = result.get_forecast(steps=horizon)
    bounds = np.asarray(fc.conf_int(alpha=0.05))
    return np.asarray(fc.predicted_mean), bounds[:, 0], bounds[:, 1]


@lru_cache(maxsize=1)
def _load_tft(key: tuple):
    import torch
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    torch.set_num_threads(4)
    # Encoders/scalers are Python objects: trusted local artifacts only.
    bundle = torch.load(key[0], map_location="cpu", weights_only=False)
    validate_metadata(bundle, "tft")
    quantiles = bundle["quantiles"]
    if sorted(set(quantiles)) != quantiles or not {0.05, 0.5, 0.95}.issubset(quantiles):
        raise ModelUnavailable("Invalid TFT quantiles. Retrain it.")
    dataset = TimeSeriesDataSet.from_parameters(bundle["dataset_parameters"], bundle["dataset_sample"])
    if dataset.max_prediction_length != MAX_HORIZON:
        raise ModelUnavailable("TFT artifact does not support the API forecast horizon. Retrain it.")
    model = TemporalFusionTransformer.from_dataset(dataset, **bundle["model_config"], loss=QuantileLoss(quantiles=bundle["quantiles"]))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, dataset, bundle["metadata"], bundle["quantiles"]


def load_tft(path: Path = TFT_PATH):
    try:
        with MODEL_LOCK:
            return _load_tft(signature(path))
    except ModelUnavailable:
        raise
    except Exception as exc:
        raise ModelUnavailable("TFT is unavailable or incompatible. Run python train_tft.py.") from exc


def predict_tft(loaded, history, horizon):
    import torch
    from pytorch_forecasting import TimeSeriesDataSet

    model, dataset, _, quantiles = loaded
    frame = prediction_frame(history, dataset.max_encoder_length, dataset.max_prediction_length)
    pred_dataset = TimeSeriesDataSet.from_dataset(dataset, frame, predict=True, stop_randomization=True)
    loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    with MODEL_LOCK, torch.inference_mode():
        x, _ = next(iter(loader))
        predictions = model(x).prediction[0].cpu().numpy()
    if predictions.shape[0] < horizon:
        raise ModelUnavailable("TFT output is shorter than the requested horizon.")
    # Independent quantile heads may cross; rearrange to produce ordered bounds.
    predictions = np.sort(predictions[:horizon], axis=-1)
    return tuple(predictions[:, quantiles.index(q)] for q in [0.5, 0.05, 0.95])


def validate_prediction(values, horizon):
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"Horizon must be between 1 and {MAX_HORIZON}.")
    arrays = tuple(np.asarray(v, dtype=float) for v in values)
    if any(v.shape != (horizon,) or not np.isfinite(v).all() for v in arrays):
        raise ModelUnavailable("Model produced invalid forecast values.")
    if np.any(arrays[1] > arrays[0]) or np.any(arrays[0] > arrays[2]):
        raise ModelUnavailable("Model produced an invalid uncertainty interval.")
    return arrays
