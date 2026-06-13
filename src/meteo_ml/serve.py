from __future__ import annotations

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from meteo_ml.data import load_dataset
from meteo_ml.features import FEATURE_NAMES, apply_standardization, standardize, to_feature_matrix
from meteo_ml.kerala import find_station, station_names
from meteo_ml.model import PrecipitationRegressor


def _bootstrap_model() -> tuple[PrecipitationRegressor, np.ndarray, np.ndarray]:
    ds = load_dataset(None)
    x_raw, y_raw = to_feature_matrix(ds)
    x_scaled, mean, std = standardize(x_raw)
    x = torch.tensor(x_scaled)
    y = torch.tensor(y_raw)

    model = PrecipitationRegressor(n_features=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.L1Loss()
    for _ in range(25):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, mean, std


app = FastAPI(title="Kerala Rainfall Forecast Postprocessing Prototype")
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
model, feature_mean, feature_std = _bootstrap_model()


class ForecastFeatures(BaseModel):
    station: str = Field("Kochi", description="Kerala station/district, for example Kochi, Idukki, Wayanad, or Kozhikode")
    day_of_year: int = Field(180, ge=1, le=366, description="Forecast valid day of year")
    temperature: float = Field(..., description="2m temperature in K")
    humidity: float = Field(..., ge=0.0, le=1.0, description="Relative humidity as fraction")
    pressure: float = Field(..., description="Surface pressure in hPa")
    wind_u: float = Field(..., description="Zonal wind component")
    wind_v: float = Field(..., description="Meridional wind component")
    nwp_precipitation: float = Field(..., ge=0.0, description="Raw NWP precipitation forecast in mm/day")


class Prediction(BaseModel):
    station: str
    district: str
    precipitation_mm: float
    heavy_rain_risk: str
    model_version: str = "kerala-prototype-0.2.0"


def _risk_label(precipitation_mm: float) -> str:
    if precipitation_mm >= 64.5:
        return "very-heavy"
    if precipitation_mm >= 20.0:
        return "heavy"
    if precipitation_mm >= 2.5:
        return "rain"
    return "light-or-dry"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "region": "Kerala"}


@app.get("/stations")
def stations() -> dict[str, list[str]]:
    return {"stations": station_names()}


@app.post("/predict", response_model=Prediction)
def predict(features: ForecastFeatures) -> Prediction:
    station = find_station(features.station)
    raw_by_name = {
        "latitude": station.latitude,
        "longitude": station.longitude,
        "elevation_m": station.elevation_m,
        "coast_distance_km": station.coast_distance_km,
        "day_of_year": float(features.day_of_year),
        "temperature": features.temperature,
        "humidity": features.humidity,
        "pressure": features.pressure,
        "wind_u": features.wind_u,
        "wind_v": features.wind_v,
        "nwp_precipitation": features.nwp_precipitation,
    }
    raw = np.array([[raw_by_name[name] for name in FEATURE_NAMES]], dtype="float32")
    x = torch.tensor(apply_standardization(raw, feature_mean, feature_std))
    with torch.no_grad():
        precipitation = float(model(x).item())
    return Prediction(
        station=station.name,
        district=station.district,
        precipitation_mm=round(precipitation, 2),
        heavy_rain_risk=_risk_label(precipitation),
    )
