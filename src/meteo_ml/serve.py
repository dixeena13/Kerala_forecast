from __future__ import annotations

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from meteo_ml.model import PrecipitationRegressor


app = FastAPI(title="ML Precipitation Postprocessing Prototype")
model = PrecipitationRegressor()
model.eval()


class ForecastFeatures(BaseModel):
    temperature: float = Field(..., description="2m temperature in K")
    humidity: float = Field(..., ge=0.0, le=1.0, description="Relative humidity as fraction")
    pressure: float = Field(..., description="Surface pressure in hPa")
    wind_u: float = Field(..., description="Zonal wind component")
    wind_v: float = Field(..., description="Meridional wind component")


class Prediction(BaseModel):
    precipitation_mm: float
    model_version: str = "prototype-0.1.0"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(features: ForecastFeatures) -> Prediction:
    raw = np.array(
        [[features.temperature, features.humidity, features.pressure, features.wind_u, features.wind_v]],
        dtype="float32",
    )
    # Placeholder normalization constants for the synthetic demo.
    mean = np.array([[294.0, 0.72, 1008.0, 0.0, 0.0]], dtype="float32")
    std = np.array([[5.0, 0.16, 8.0, 4.0, 4.0]], dtype="float32")
    x = torch.tensor((raw - mean) / std)
    with torch.no_grad():
        precipitation = float(model(x).item())
    return Prediction(precipitation_mm=round(precipitation, 3))

