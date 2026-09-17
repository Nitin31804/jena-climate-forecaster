import numpy as np
import pytest

from backtest import evaluation_origins, metrics
from forecast_api.config import DATE
from forecast_api.data import split_boundaries


def metadata(history):
    train, val = split_boundaries(len(history))
    return {"data_fingerprint": "test", "trained_through": history[DATE].iloc[train - 1].isoformat(),
            "validation_through": history[DATE].iloc[val - 1].isoformat(),
            "test_start": history[DATE].iloc[val].isoformat()}


def test_origins_are_held_out_and_disjoint(history):
    origins = evaluation_origins(history, [metadata(history)], "test", 24, 5)
    _, end = split_boundaries(len(history))
    assert min(origins) >= end
    assert np.diff(origins).min() >= 24
    assert max(origins) + 24 <= len(history)


@pytest.mark.parametrize("field", ["trained_through", "validation_through", "data_fingerprint", "test_start"])
def test_leaking_or_mismatched_artifact_rejected(history, field):
    meta = metadata(history)
    meta[field] = "different" if field == "data_fingerprint" else history[DATE].iloc[-1].isoformat()
    with pytest.raises(ValueError):
        evaluation_origins(history, [meta], "test", 24, 5)


def test_imputed_targets_are_not_scored(history):
    _, start = split_boundaries(len(history))
    history.loc[start:start + 23, "temperature_observed"] = False
    origins = evaluation_origins(history, [metadata(history)], "test", 24, 5)
    assert start not in origins


def test_metrics_work_with_zero_and_negative_celsius():
    assert metrics([0, -2, 2], [1, -1, 3]) == {"mae_c": 1.0, "rmse_c": 1.0}


def test_metrics_reject_accidental_broadcasting():
    with pytest.raises(ValueError):
        metrics([1, 2, 3], [1])
