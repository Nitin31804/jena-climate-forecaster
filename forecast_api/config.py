"""Shared paths, feature definitions, and modeling defaults."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("JENA_DATA_PATH", str(ROOT / "df_hourly.csv")))
ARTIFACT_DIR = Path(os.getenv("JENA_ARTIFACT_DIR", str(ROOT / "artifacts")))
SARIMA_PATH = ARTIFACT_DIR / "sarima.json"
TFT_PATH = ARTIFACT_DIR / "tft.pt"
TARGET = "T (degC)"
DATE = "Date Time"
ENCODER_LENGTH = 168
MAX_HORIZON = 168
SARIMA_CONTEXT = 720
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
SEED = 42
FORMAT_VERSION = 1
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
KNOWN_FEATURES = ["day_sin", "day_cos", "year_sin", "year_cos"]
UNKNOWN_FEATURES = [TARGET, "temperature_mean_6h", "temperature_mean_24h"]
SARIMA_SPEC = {"order": (1, 1, 1), "seasonal_order": (1, 0, 1, 24)}
TFT_SPEC = {"learning_rate": 0.001, "hidden_size": 16, "attention_head_size": 1,
            "dropout": 0.1, "hidden_continuous_size": 8}
