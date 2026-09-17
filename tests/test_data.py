import io
import json
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

from export_hourly import preprocess
from forecast_api import data
from forecast_api.config import DATE, TARGET, UNKNOWN_FEATURES


def test_future_rows_start_after_observations(history):
    frame = data.prediction_frame(history)
    assert len(frame) == 336
    assert frame[DATE].iloc[168] == history[DATE].iloc[-1] + pd.Timedelta(hours=1)
    assert frame[DATE].iloc[-1] == history[DATE].iloc[-1] + pd.Timedelta(hours=168)
    assert frame.time_idx.diff().iloc[1:].eq(1).all()
    for column in UNKNOWN_FEATURES:
        assert frame[column].iloc[168:].eq(frame[column].iloc[167]).all()


def test_features_do_not_use_future_values(history):
    before = data.features(history).iloc[:500]
    history.loc[600:, TARGET] = -1000
    pd.testing.assert_frame_equal(before, data.features(history).iloc[:500])


@pytest.mark.parametrize("problem", ["gap", "duplicate", "nan", "infinity", "short"])
def test_invalid_history_rejected(history, problem):
    if problem == "gap":
        history = history.drop(index=100)
    elif problem == "duplicate":
        history.loc[100, DATE] = history.loc[99, DATE]
    elif problem == "short":
        history = history.head(100)
    else:
        history.loc[100, TARGET] = np.nan if problem == "nan" else np.inf
    with pytest.raises(data.DataUnavailable):
        data.prediction_frame(history)


def test_preprocess_hour_end_and_missing_observation_mask():
    dates = pd.date_range("2010-01-01 00:10", periods=40 * 6, freq="10min")
    raw = pd.DataFrame({DATE: dates.strftime("%d.%m.%Y %H:%M:%S"), TARGET: np.arange(len(dates), dtype=float)})
    # Keep a gap after rolling-feature warmup, then a large future jump.
    raw = raw.loc[~((dates > "2010-01-02 02:00") & (dates <= "2010-01-02 06:00"))]
    result = preprocess(raw)
    assert not result.loc["2010-01-02 03:00", "temperature_observed"]
    assert result.loc["2010-01-02 03:00", TARGET] == result.loc["2010-01-02 02:00", TARGET]
    assert result.loc["2010-01-02 01:00", TARGET] == np.mean(np.arange(144, 150))


def test_live_request_filters_future_and_caches(monkeypatch):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    dates = pd.date_range(now - pd.Timedelta(hours=220), periods=245, freq="h")
    payload = json.dumps({"hourly": {"time": [t.isoformat() for t in dates], "temperature_2m": [10.0] * len(dates)}}).encode()
    calls = []

    def request(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(data, "urlopen", request)
    monkeypatch.setattr(data, "_live_cache", None)
    first = data.live_history()
    second = data.live_history()
    assert first[DATE].iloc[-1] == now
    assert len(calls) == 1 and calls[0][1] == 10
    pd.testing.assert_frame_equal(first, second)


def test_upstream_failure_is_actionable(monkeypatch):
    monkeypatch.setattr(data, "_live_cache", None)

    def fail(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(data, "urlopen", fail)
    with pytest.raises(data.LiveDataUnavailable, match="historical mode"):
        data.live_history()
