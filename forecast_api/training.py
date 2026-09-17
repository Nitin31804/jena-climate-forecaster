"""Common provenance and dataset construction for repeatable experiments."""
from datetime import datetime, timezone
from importlib.metadata import version
from uuid import uuid4

from .config import DATE, ENCODER_LENGTH, KNOWN_FEATURES, MAX_HORIZON, SEED, TARGET, UNKNOWN_FEATURES
from .data import features, split_boundaries


def provenance(model, history, fingerprint):
    train_end, val_end = split_boundaries(len(history))
    packages = ["numpy", "pandas", "scipy", "statsmodels"]
    if model == "tft":
        packages += ["torch", "lightning", "pytorch-forecasting", "scikit-learn"]
    return {
        "model": model, "version": f"{model}-{uuid4().hex[:12]}",
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "data_fingerprint": fingerprint, "training_rows": train_end,
        "trained_through": history[DATE].iloc[train_end - 1].isoformat(),
        "validation_through": history[DATE].iloc[val_end - 1].isoformat(),
        "test_start": history[DATE].iloc[val_end].isoformat(),
        "data_timezone": "Dataset local time (offset unspecified)",
        "dependencies": {name: version(name) for name in packages},
    }


def tft_datasets(history):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    train_end, val_end = split_boundaries(len(history))
    frame = features(history.iloc[:val_end])
    train_frame = frame.loc[frame.time_idx < train_end].copy()
    training = TimeSeriesDataSet(
        train_frame, time_idx="time_idx", target=TARGET, group_ids=["series_id"],
        max_encoder_length=ENCODER_LENGTH, max_prediction_length=MAX_HORIZON,
        time_varying_known_reals=KNOWN_FEATURES,
        time_varying_unknown_reals=UNKNOWN_FEATURES,
        target_normalizer=GroupNormalizer(groups=["series_id"]),
    )
    # Encoder context may precede the boundary; every validation target must follow it.
    validation = TimeSeriesDataSet.from_dataset(
        training, frame.loc[frame.time_idx >= train_end - ENCODER_LENGTH],
        min_prediction_idx=train_end, stop_randomization=True,
    )
    if "temperature_observed" in history:
        import numpy as np
        missing = np.r_[0, (~history["temperature_observed"]).astype(int).cumsum()]

        def observed_targets(index):
            return missing[index.time_idx_last.to_numpy() + 1] == missing[index.time_idx_first_prediction.to_numpy()]

        training = training.filter(observed_targets)
        validation = validation.filter(observed_targets)
    return training, validation, train_frame
