"""Forecast orchestration and a bounded cache for repeated requests."""
from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock

import pandas as pd

from .config import DATE, TARGET
from .data import historical_data, live_history
from .models import ModelUnavailable, load_sarima, load_tft, predict_sarima, predict_tft, validate_prediction

_cache = OrderedDict()
_lock = Lock()


def forecast(model: str, horizon: int, live: bool) -> dict:
    history, digest = (live_history(), None) if live else historical_data()
    loaded = load_sarima() if model == "sarima" else load_tft()
    metadata = loaded["metadata"] if model == "sarima" else loaded[2]
    if not live and metadata["data_fingerprint"] != digest:
        raise ModelUnavailable("Historical data has changed since training. Retrain the model.")
    cutoff = history[DATE].iloc[-1]
    history_hash = int(pd.util.hash_pandas_object(history.tail(720), index=False).sum())
    key = (model, metadata["version"], live, cutoff.isoformat(), history_hash, horizon)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return {**_cache[key], "timestamp": datetime.now(timezone.utc)}
        function = predict_sarima if model == "sarima" else predict_tft
        median, lower, upper = validate_prediction(function(loaded, history, horizon), horizon)
        result = {
            "city": "Jena", "model": model, "horizon": horizon,
            "forecast": median.tolist(), "lower_ci": lower.tolist(), "upper_ci": upper.tolist(),
            "forecast_timestamps": [t.isoformat() for t in pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")],
            "data_cutoff": cutoff.isoformat(),
            "timezone": "UTC" if live else "Dataset local time (offset unspecified)",
            "mode": "live" if live else "historical",
            "data_source": "Open-Meteo model-derived weather" if live else "Jena Climate station dataset",
            "model_version": metadata["version"], "trained_through": metadata["trained_through"],
            "interval_level": 0.95 if model == "sarima" else 0.90,
            "interval_method": "SARIMA model interval" if model == "sarima" else "TFT 5th–95th quantiles (ordered)",
            "history_timestamps": [t.isoformat() for t in history[DATE].tail(48)],
            "history": history[TARGET].tail(48).tolist(),
            "timestamp": datetime.now(timezone.utc),
        }
        _cache[key] = result
        if len(_cache) > 32:
            _cache.popitem(last=False)
        return dict(result)
