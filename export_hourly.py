"""Download Jena observations and produce causal hourly features."""
import argparse
import os
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import shutil

import numpy as np
import pandas as pd

from forecast_api.config import DATA_PATH, DATE, ROOT, TARGET
from forecast_api.data import features, validate_history

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"


def preprocess(raw):
    raw = raw.copy()
    raw[DATE] = pd.to_datetime(raw[DATE], format="%d.%m.%Y %H:%M:%S", errors="raise")
    numeric = raw.set_index(DATE).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.mask(numeric <= -999)
    # Label the hour by its END: all observations are available by its timestamp.
    hourly = numeric.resample("h", closed="right", label="right").mean()
    observed = hourly[TARGET].notna()
    # Causal imputation keeps an hourly grid without borrowing future observations.
    # Preserve a mask so imputed temperatures never become evaluation targets.
    hourly = hourly.ffill()
    base = validate_history(hourly.reset_index())
    engineered = features(base).set_index(DATE)
    result = hourly.loc[engineered.index].copy()
    result["temperature_observed"] = observed.loc[engineered.index]
    for col in ["temperature_mean_6h", "temperature_mean_24h", "day_sin", "day_cos", "year_sin", "year_cos"]:
        result[col] = engineered[col]
    if not np.isfinite(result[TARGET]).all():
        raise ValueError("Temperature gaps remain after limited forward filling.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, help="Use an existing raw CSV instead of downloading")
    args = parser.parse_args()
    if args.raw_csv:
        raw = pd.read_csv(args.raw_csv)
    else:
        archive = ROOT / "jena_climate_2009_2016.csv.zip"
        if not archive.exists():
            temporary = archive.with_suffix(".download")
            try:
                with urlopen(URL, timeout=30) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                os.replace(temporary, archive)
            finally:
                temporary.unlink(missing_ok=True)
        with ZipFile(archive) as zipped, zipped.open("jena_climate_2009_2016.csv") as csv:
            raw = pd.read_csv(csv)
    hourly = preprocess(raw)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = DATA_PATH.with_suffix(".csv.tmp")
    hourly.to_csv(temporary)
    os.replace(temporary, DATA_PATH)
    print(f"Saved {len(hourly):,} hourly rows to {DATA_PATH}. Retrain both models after changing data.")


if __name__ == "__main__":
    main()
