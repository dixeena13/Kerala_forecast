from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

os.environ.setdefault("MPLCONFIGDIR", "/tmp/meteo-ml-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import truststore

from meteo_ml.kerala import KERALA_STATIONS

truststore.inject_into_ssl()


def fetch_daily_rain(latitude: float, longitude: float) -> tuple[str, float, float]:
    params = {
        "latitude": round(latitude, 3),
        "longitude": round(longitude, 3),
        "daily": "precipitation_sum,precipitation_probability_max",
        "timezone": "Asia/Kolkata",
        "forecast_days": 1,
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    daily = payload["daily"]
    return daily["time"][0], float(daily["precipitation_sum"][0]), float(daily["precipitation_probability_max"][0])


def main() -> None:
    output = Path("figures/kerala_real_rainfall_field_map.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Lightweight Kerala bounding-box grid. This approximates a weather-field map
    # without requiring heavy GIS or GRIB processing dependencies.
    lat_values = np.linspace(8.3, 12.1, 12)
    lon_values = np.linspace(75.1, 77.3, 10)
    rain_grid = np.zeros((len(lat_values), len(lon_values)), dtype="float32")
    probability_grid = np.zeros_like(rain_grid)
    valid_date = ""

    for i, lat in enumerate(lat_values):
        for j, lon in enumerate(lon_values):
            valid_date, rain, probability = fetch_daily_rain(lat, lon)
            rain_grid[i, j] = rain
            probability_grid[i, j] = probability

    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    valid_label = datetime.fromisoformat(valid_date).strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(8.5, 9.5), constrained_layout=True)
    levels = np.linspace(0.0, max(1.0, float(np.nanmax(rain_grid))), 12)
    field = ax.contourf(lon_grid, lat_grid, rain_grid, levels=levels, cmap="YlGnBu", extend="max")
    ax.contour(lon_grid, lat_grid, probability_grid, levels=[30, 60, 80], colors="#374151", linewidths=0.8, alpha=0.55)

    station_lons = [station.longitude for station in KERALA_STATIONS]
    station_lats = [station.latitude for station in KERALA_STATIONS]
    ax.scatter(station_lons, station_lats, s=28, color="#111827", edgecolor="white", linewidth=0.8, zorder=5)

    for station in KERALA_STATIONS:
        ax.annotate(
            station.name,
            (station.longitude, station.latitude),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
            color="#111827",
            zorder=6,
        )

    colorbar = fig.colorbar(field, ax=ax, shrink=0.8, pad=0.03)
    colorbar.set_label("Forecast rainfall (mm/day)")

    ax.set_title(f"Real Forecast Rainfall Field: Kerala ({valid_label})", fontsize=15, pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(75.0, 77.4)
    ax.set_ylim(8.1, 12.25)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", alpha=0.3, linewidth=0.7)

    ax.text(
        0.02,
        0.02,
        "Filled field: rainfall. Thin contours: rain probability (%). Source: Open-Meteo Forecast API.",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.88},
    )

    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
