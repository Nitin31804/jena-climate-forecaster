import numpy as np
import pandas as pd
import pytest

from forecast_api.config import DATE, TARGET


@pytest.fixture
def history():
    t = np.arange(2400)
    return pd.DataFrame({DATE: pd.date_range("2010-01-01", periods=len(t), freq="h"),
                         TARGET: 8 + 5 * np.sin(t * 2 * np.pi / 24) + t / 10000,
                         "temperature_observed": True})
