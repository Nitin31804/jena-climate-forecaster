"""Evaluate frozen models and naive baselines at disjoint held-out forecast origins."""
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from forecast_api.config import DATE, ROOT, TARGET
from forecast_api.data import historical_data, split_boundaries
from forecast_api.models import atomic_json, load_sarima, load_tft, predict_sarima, predict_tft, validate_prediction


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Actuals and predictions must have identical one-dimensional shapes.")
    error = actual - predicted
    if error.ndim != 1 or not len(error) or not np.isfinite(error).all():
        raise ValueError("Evaluation requires finite, aligned one-dimensional arrays.")
    return {"mae_c": float(np.mean(np.abs(error))), "rmse_c": float(np.sqrt(np.mean(error ** 2)))}


def evaluation_origins(history, metadata, digest, horizon, windows):
    _, val_end = split_boundaries(len(history))
    for meta in metadata:
        if meta["data_fingerprint"] != digest:
            raise ValueError("Artifact data differs from evaluation data; retrain before evaluating.")
        if pd.Timestamp(meta["validation_through"]) >= history[DATE].iloc[val_end]:
            raise ValueError("Evaluation would overlap model validation or training data.")
        if pd.Timestamp(meta["trained_through"]) >= history[DATE].iloc[val_end]:
            raise ValueError("Evaluation would overlap model training data.")
        if pd.Timestamp(meta["test_start"]) != history[DATE].iloc[val_end]:
            raise ValueError("Artifact split differs from the current test split.")
    candidates = list(range(val_end, len(history) - horizon + 1, horizon))
    if "temperature_observed" in history:
        candidates = [i for i in candidates if history["temperature_observed"].iloc[i:i + horizon].all()]
    available = len(candidates)
    if windows < 1 or windows > available:
        raise ValueError(f"Choose between 1 and {available} disjoint windows.")
    # Evenly sample the entire held-out period; no overlapping targets.
    return [candidates[int(i)] for i in np.linspace(0, available - 1, windows, dtype=int)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--models", nargs="+", choices=["sarima", "tft"], default=["sarima", "tft"])
    args = parser.parse_args()
    if not 1 <= args.horizon <= 168:
        parser.error("--horizon must be between 1 and 168")
    history, digest = historical_data()
    loaded = {name: load_sarima() if name == "sarima" else load_tft() for name in dict.fromkeys(args.models)}
    metadata = {name: value["metadata"] if name == "sarima" else value[2] for name, value in loaded.items()}
    origins = evaluation_origins(history, list(metadata.values()), digest, args.horizon, args.windows)
    rows, errors = [], {}
    for origin in origins:
        context = history.iloc[:origin]
        actual = history[TARGET].iloc[origin:origin + args.horizon].to_numpy()
        predictions = {
            "Persistence": (np.repeat(context[TARGET].iloc[-1], args.horizon), None, None),
            "Seasonal naive (24h)": (np.resize(context[TARGET].iloc[-24:].to_numpy(), args.horizon), None, None),
        }
        for name, model in loaded.items():
            fn = predict_sarima if name == "sarima" else predict_tft
            predictions[name.upper()] = validate_prediction(fn(model, context, args.horizon), args.horizon)
        for name, (point, lower, upper) in predictions.items():
            row = {"model": name, "forecast_start": history[DATE].iloc[origin].isoformat(), **metrics(actual, point)}
            if lower is not None:
                row["interval_coverage"] = float(np.mean((actual >= lower) & (actual <= upper)))
            rows.append(row)
            errors.setdefault(name, []).extend((actual - point).tolist())
        print(f"Evaluated {history[DATE].iloc[origin]} ({args.horizon} hours)")
    summary = [{"model": name, **metrics(np.asarray(error), np.zeros(len(error)))} for name, error in errors.items()]
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "data_fingerprint": digest,
              "horizon": args.horizon, "windows": args.windows, "artifacts": metadata,
              "method": "Frozen parameters; expanding available history; disjoint held-out target windows",
              "summary": summary, "results": rows}
    destination = ROOT / "assets" / "backtest_results.json"
    atomic_json(destination, report)
    table = pd.DataFrame(summary).rename(columns={"model": "Model", "mae_c": "MAE (°C)", "rmse_c": "RMSE (°C)"})
    text = ("# Held-out temperature evaluation\n\n"
            f"Generated {report['generated_at']}. {args.windows} disjoint windows, {args.horizon} hours each.\n\n"
            "Models use frozen parameters; all targets follow training and validation. "
            "SARIMA conditions on up to 720 past hours; TFT uses 168 past hours. "
            "Only calendar covariates are supplied for future hours.\n\n"
            + table.to_markdown(index=False, floatfmt=".3f") + "\n\n"
            "Lower is better. These sampled windows do not establish general superiority. "
            "TFT training settings, model versions, per-window errors, and interval coverage "
            "are recorded in backtest_results.json. Celsius percentage errors are intentionally omitted.\n")
    destination.with_suffix(".md").write_text(text, encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
