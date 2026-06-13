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


def fetch_station_daily_forecast(latitude: float, longitude: float) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "precipitation_sum,precipitation_probability_max",
        "timezone": "Asia/Kolkata",
        "forecast_days": 1,
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    output = Path("figures/kerala_real_station_forecast_map.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for station in KERALA_STATIONS:
        payload = fetch_station_daily_forecast(station.latitude, station.longitude)
        daily = payload["daily"]
        rows.append(
            {
                "name": station.name,
                "lat": station.latitude,
                "lon": station.longitude,
                "rain": float(daily["precipitation_sum"][0]),
                "probability": float(daily["precipitation_probability_max"][0]),
                "date": daily["time"][0],
            }
        )

    lats = np.array([row["lat"] for row in rows])
    lons = np.array([row["lon"] for row in rows])
    rain = np.array([row["rain"] for row in rows])
    probability = np.array([row["probability"] for row in rows])
    sizes = 90 + 3.0 * probability
    valid_date = datetime.fromisoformat(rows[0]["date"]).strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
    scatter = ax.scatter(
        lons,
        lats,
        c=rain,
        s=sizes,
        cmap="YlGnBu",
        edgecolor="#1f2937",
        linewidth=0.8,
        alpha=0.95,
    )

    for row in rows:
        ax.annotate(
            row["name"],
            (row["lon"], row["lat"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.78, pad=0.03)
    colorbar.set_label("Forecast rainfall (mm/day)")

    ax.set_title(f"Real Forecast Station Map: Kerala Rainfall ({valid_date})", fontsize=14, pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(75.0, 77.4)
    ax.set_ylim(8.1, 12.2)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    ax.text(
        0.02,
        0.02,
        "Circle size: max rain probability. Source: Open-Meteo Forecast API.",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.85},
    )

    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
