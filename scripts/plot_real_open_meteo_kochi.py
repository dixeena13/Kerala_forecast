from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

os.environ.setdefault("MPLCONFIGDIR", "/tmp/meteo-ml-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import truststore

truststore.inject_into_ssl()

KOCHI_LAT = 9.9312
KOCHI_LON = 76.2673


def fetch_kochi_forecast() -> dict:
    params = {
        "latitude": KOCHI_LAT,
        "longitude": KOCHI_LON,
        "daily": "precipitation_sum,precipitation_probability_max",
        "timezone": "Asia/Kolkata",
        "forecast_days": 7,
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    output = Path("figures/kochi_real_open_meteo_forecast.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = fetch_kochi_forecast()
    daily = payload["daily"]
    dates = [datetime.fromisoformat(day) for day in daily["time"]]
    rain = daily["precipitation_sum"]
    probability = daily["precipitation_probability_max"]

    fig, ax_rain = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax_prob = ax_rain.twinx()

    ax_rain.bar(dates, rain, color="#277da1", alpha=0.82, label="forecast rainfall")
    ax_prob.plot(dates, probability, color="#d62828", marker="o", linewidth=2.0, label="rain probability")

    ax_rain.set_title("Real forecast data: Kochi daily rainfall from Open-Meteo")
    ax_rain.set_ylabel("Precipitation sum (mm/day)")
    ax_prob.set_ylabel("Max precipitation probability (%)")
    ax_rain.set_xlabel("Forecast date")
    ax_rain.grid(axis="y", alpha=0.25)
    ax_rain.tick_params(axis="x", rotation=30)

    lines, labels = ax_rain.get_legend_handles_labels()
    lines2, labels2 = ax_prob.get_legend_handles_labels()
    ax_rain.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.text(
        0.01,
        0.01,
        "Source: Open-Meteo Forecast API, Kochi, Kerala.",
        fontsize=8,
    )
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
