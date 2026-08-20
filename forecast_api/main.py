# forecast_api/main.py
"""FastAPI application exposing a temperature forecast endpoint.

Endpoint:
    GET /forecast?city=Jena&horizon=24&model=tft

Parameters
    city (str): Currently only "Jena" is supported (dataset is Jena climate).
    horizon (int): Number of future hours to predict (default 24, max 168).
    model (str): Which model to use - "tft" or "sarima". Default is "sarima".

Response (JSON):
    {
        "city": "Jena",
        "model": "sarima",
        "horizon": 24,
        "forecast": [list of point predictions],
        "lower_ci": [list of lower bound],
        "upper_ci": [list of upper bound],
        "timestamp": "ISO-8601 time when forecast was generated"
    }
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import datetime

from .utils import load_sarima_model, load_tft_model, forecast_sarima, forecast_tft

app = FastAPI(title="Weather Forecast API", version="0.1.0")


class ForecastResponse(BaseModel):
    city: str
    model: str
    horizon: int
    forecast: List[float]
    lower_ci: Optional[List[float]] = None
    upper_ci: Optional[List[float]] = None
    timestamp: datetime.datetime


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    city: str = Query("Jena", description="City name (only Jena supported for demo)"),
    horizon: int = Query(24, ge=1, le=168, description="Number of hours to forecast"),
    model: str = Query("sarima", description="Model to use: sarima or tft"),
    use_live_data: bool = Query(False, description="Fetch live weather data for Jena from open-meteo"),
):
    if city.lower() != "jena":
        raise HTTPException(status_code=400, detail="Only Jena dataset is available in this demo.")

    if model not in ("sarima", "tft"):
        raise HTTPException(status_code=400, detail="Model must be 'sarima' or 'tft'.")

    if model == "sarima":
        sarima = load_sarima_model()
        preds, lower, upper = forecast_sarima(sarima, horizon, use_live_data)
        return ForecastResponse(
            city=city,
            model=model,
            horizon=horizon,
            forecast=preds,
            lower_ci=lower,
            upper_ci=upper,
            timestamp=datetime.datetime.utcnow(),
        )
    elif model == "tft":
        try:
            tft, dataset = load_tft_model()
            preds, lower, upper = forecast_tft(tft, dataset, horizon, use_live_data)
            return ForecastResponse(
                city=city,
                model=model,
                horizon=horizon,
                forecast=preds,
                lower_ci=lower,
                upper_ci=upper,
                timestamp=datetime.datetime.utcnow(),
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail="TFT inference error: " + str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
