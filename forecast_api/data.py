"""Causal feature preparation shared by training, evaluation, and inference."""
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import (DATA_PATH, DATE, ENCODER_LENGTH, KNOWN_FEATURES, MAX_HORIZON,
                     TARGET, TRAIN_FRACTION, UNKNOWN_FEATURES, VALIDATION_FRACTION)


class DataUnavailable(RuntimeError):
    """Local data is missing or invalid."""


class LiveDataUnavailable(RuntimeError):
    """The upstream provider could not supply usable recent weather."""


def signature(path: Path) -> tuple:
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise DataUnavailable(f"{path.name} is missing. Run the preparation command in README.md.") from exc
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


def validate_history(frame: pd.DataFrame, minimum: int = 1) -> pd.DataFrame:
    if not {DATE, TARGET}.issubset(frame):
        raise DataUnavailable(f"Data must contain {DATE!r} and {TARGET!r}.")
    cols = [DATE, TARGET] + (["temperature_observed"] if "temperature_observed" in frame else [])
    df = frame[cols].copy().reset_index(drop=True)
    if "temperature_observed" in df:
        if not df["temperature_observed"].isin([True, False]).all():
            raise DataUnavailable("temperature_observed must contain boolean values.")
    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if len(df) < minimum or df[DATE].isna().any() or not np.isfinite(df[TARGET]).all():
        raise DataUnavailable(f"Need at least {minimum} valid hourly temperature rows.")
    if len(df) > 1 and not df[DATE].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise DataUnavailable("Temperature timestamps must be unique, ordered, and exactly one hour apart.")
    return df


@lru_cache(maxsize=2)
def _read_history(key: tuple) -> tuple[pd.DataFrame, str]:
    try:
        frame = validate_history(pd.read_csv(key[0]))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataUnavailable("Historical data cannot be read. Run python export_hourly.py.") from exc
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()
    return frame, digest


def historical_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, str]:
    frame, digest = _read_history(signature(path))
    return frame.copy(), digest


def split_boundaries(length: int) -> tuple[int, int]:
    train_end = int(length * TRAIN_FRACTION)
    val_end = int(length * (TRAIN_FRACTION + VALIDATION_FRACTION))
    if train_end < ENCODER_LENGTH + MAX_HORIZON + 23 or val_end - train_end < MAX_HORIZON or length - val_end < MAX_HORIZON:
        raise DataUnavailable("Dataset is too short for disjoint training, validation, and test periods.")
    return train_end, val_end


def calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    dates = pd.to_datetime(df[DATE])
    day_phase = (dates.dt.hour + dates.dt.minute / 60) / 24
    year_phase = (dates.dt.dayofyear - 1 + day_phase) / 365.2425
    for name, phase in [("day", day_phase), ("year", year_phase)]:
        df[f"{name}_sin"] = np.sin(2 * np.pi * phase)
        df[f"{name}_cos"] = np.cos(2 * np.pi * phase)
    return df


def features(history: pd.DataFrame) -> pd.DataFrame:
    df = calendar_features(validate_history(history, 24))
    df["time_idx"] = np.arange(len(df), dtype=np.int64)
    df["series_id"] = "jena"
    df["temperature_mean_6h"] = df[TARGET].rolling(6).mean()
    df["temperature_mean_24h"] = df[TARGET].rolling(24).mean()
    return df.iloc[23:].reset_index(drop=True)


def prediction_frame(history: pd.DataFrame, encoder_length: int = ENCODER_LENGTH,
                     horizon: int = MAX_HORIZON) -> pd.DataFrame:
    """Append decoder rows strictly AFTER history; never consume future targets."""
    validate_history(history, encoder_length + 23)
    encoder = features(history).iloc[-encoder_length:].copy()
    future = pd.DataFrame({DATE: pd.date_range(encoder[DATE].iloc[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")})
    future = calendar_features(future)
    future["time_idx"] = np.arange(encoder["time_idx"].iloc[-1] + 1, encoder["time_idx"].iloc[-1] + horizon + 1)
    future["series_id"] = "jena"
    # Unknown reals are excluded from TFT decoder inputs. These placeholders only
    # satisfy the dataset schema; calendar features are the future inputs.
    for col in UNKNOWN_FEATURES:
        future[col] = encoder[col].iloc[-1]
    cols = [DATE, "time_idx", "series_id", *KNOWN_FEATURES, *UNKNOWN_FEATURES]
    return pd.concat([encoder[cols], future[cols]], ignore_index=True)


_live_lock = Lock()
_live_cache: tuple[float, pd.DataFrame] | None = None
LIVE_URL = ("https://api.open-meteo.com/v1/forecast?latitude=50.9272&longitude=11.5861"
            "&past_days=14&forecast_days=1&hourly=temperature_2m&timezone=UTC")


def live_history() -> pd.DataFrame:
    global _live_cache
    with _live_lock:
        if _live_cache is not None and time.monotonic() - _live_cache[0] < 300:
            return _live_cache[1].copy()
        try:
            with urlopen(LIVE_URL, timeout=10) as response:
                payload = json.load(response)
            df = pd.DataFrame({DATE: pd.to_datetime(payload["hourly"]["time"], utc=True),
                               TARGET: payload["hourly"]["temperature_2m"]})
            now = pd.Timestamp.now(tz="UTC")
            df = validate_history(df.loc[df[DATE] <= now], ENCODER_LENGTH + 23)
            if now - df[DATE].iloc[-1] > pd.Timedelta(hours=2):
                raise ValueError("Latest upstream hour is stale.")
        except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError, DataUnavailable) as exc:
            raise LiveDataUnavailable("Recent weather is unavailable. Retry shortly or use historical mode.") from exc
        _live_cache = time.monotonic(), df
        return df.copy()
