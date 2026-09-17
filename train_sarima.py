"""Fit SARIMA on training data and save compact parameters, never a result pickle."""
import argparse
import numpy as np

from forecast_api.config import FORMAT_VERSION, SARIMA_CONTEXT, SARIMA_PATH, SARIMA_SPEC
from forecast_api.data import historical_data, split_boundaries
from forecast_api.models import atomic_json, fit_sarima
from forecast_api.training import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=200)
    args = parser.parse_args()
    if args.maxiter < 1:
        parser.error("--maxiter must be positive")
    history, digest = historical_data()
    train_end, _ = split_boundaries(len(history))
    result = fit_sarima(history.iloc[:train_end], args.maxiter)
    if not result.mle_retvals.get("converged") or not np.isfinite(result.params).all():
        raise RuntimeError("SARIMA did not converge. Increase --maxiter; no artifact was replaced.")
    metadata = provenance("sarima", history, digest)
    metadata.update(converged=True, fit_context_hours=SARIMA_CONTEXT)
    atomic_json(SARIMA_PATH, {"format_version": FORMAT_VERSION, "metadata": metadata,
                             "spec": SARIMA_SPEC, "parameters": result.params.tolist()})
    print(f"Saved {SARIMA_PATH} ({SARIMA_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
