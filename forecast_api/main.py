"""Forecast API and same-origin dashboard."""
import logging
import os
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ROOT
from .data import DataUnavailable, LiveDataUnavailable, historical_data
from .models import ModelUnavailable, load_sarima, load_tft
from .utils import forecast

logger = logging.getLogger(__name__)
app = FastAPI(title="Jena Climate Forecaster", version="1.0.0")
origins = [origin.strip() for origin in os.getenv("JENA_CORS_ORIGINS", "").split(",") if origin.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


class ForecastResponse(BaseModel):
    city: str
    model: Literal["sarima", "tft"]
    horizon: int
    forecast: list[float]
    lower_ci: list[float]
    upper_ci: list[float]
    forecast_timestamps: list[str]
    data_cutoff: str
    timezone: str
    mode: Literal["historical", "live"]
    data_source: str
    model_version: str
    trained_through: str
    interval_level: float
    interval_method: str
    history_timestamps: list[str]
    history: list[float]
    timestamp: datetime


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(city: str = "Jena", horizon: int = Query(24, ge=1, le=168),
                 model: Literal["sarima", "tft"] = "sarima", use_live_data: bool = False):
    if city.strip().lower() != "jena":
        raise HTTPException(400, "Only Jena is supported.")
    try:
        return forecast(model, horizon, use_live_data)
    except (DataUnavailable, ModelUnavailable) as exc:
        raise HTTPException(503, str(exc)) from exc
    except LiveDataUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc
    except Exception as exc:
        logger.exception("Forecast failed for %s", model)
        raise HTTPException(500, "Forecast could not be generated. Check the server log.") from exc


@app.get("/health")
def health():
    """Liveness only; /ready verifies model and data availability."""
    return {"status": "ok", "version": app.version}


@app.get("/ready")
def ready(model: Literal["sarima", "tft"] = "sarima"):
    try:
        _, digest = historical_data()
        loaded = load_sarima() if model == "sarima" else load_tft()
        meta = loaded["metadata"] if model == "sarima" else loaded[2]
        if meta["data_fingerprint"] != digest:
            raise ModelUnavailable("Data changed after training. Retrain the model.")
        return {"status": "ready", "model": model, "model_version": meta["version"]}
    except (DataUnavailable, ModelUnavailable) as exc:
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": str(exc)})
