from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/meteo-ml-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meteo_ml.scenarios import (
    ensemble_change_summary,
    extreme_rainfall_indicators,
    make_extreme_rainfall_scenario_dataset,
)


def main() -> None:
    output = Path("figures/climate_extreme_scenario_demo.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    ds = make_extreme_rainfall_scenario_dataset(n_years=30, n_members=10, seed=18)
    indicators = extreme_rainfall_indicators(ds)
    summary = ensemble_change_summary(indicators)

    selected_stations = ["Kochi", "Idukki", "Wayanad", "Kozhikode"]
    x = np.arange(len(selected_stations))

    heavy_change = summary["heavy_rain_days_per_year_change_mean"].sel(station=selected_stations).values
    heavy_p10 = summary["heavy_rain_days_per_year_change_p10"].sel(station=selected_stations).values
    heavy_p90 = summary["heavy_rain_days_per_year_change_p90"].sel(station=selected_stations).values
    max_change = summary["annual_max_daily_rainfall_change_mean"].sel(station=selected_stations).values
    p95_change = summary["p95_daily_rainfall_change_mean"].sel(station=selected_stations).values

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.42})
    ax_days, ax_intensity = axes

    yerr = np.vstack([heavy_change - heavy_p10, heavy_p90 - heavy_change])
    ax_days.bar(x, heavy_change, yerr=yerr, capsize=5, color="#2a9d8f", edgecolor="#1f2937", linewidth=0.6)
    ax_days.axhline(0.0, color="#374151", linewidth=0.9)
    ax_days.set_title("Simulated future change in heavy-rain days", fontsize=14, pad=10)
    ax_days.set_ylabel("Change in days/year\n(>= 20 mm/day)")
    ax_days.set_xticks(x)
    ax_days.set_xticklabels(selected_stations)
    ax_days.grid(axis="y", alpha=0.25)

    width = 0.36
    ax_intensity.bar(x - width / 2, max_change, width=width, color="#457b9d", label="annual max daily rainfall")
    ax_intensity.bar(x + width / 2, p95_change, width=width, color="#e9c46a", label="95th percentile daily rainfall")
    ax_intensity.axhline(0.0, color="#374151", linewidth=0.9)
    ax_intensity.set_title("Simulated change in rainfall intensity indicators", fontsize=14, pad=10)
    ax_intensity.set_ylabel("Change (mm/day)")
    ax_intensity.set_xlabel("Station")
    ax_intensity.set_xticks(x)
    ax_intensity.set_xticklabels(selected_stations)
    ax_intensity.legend(fontsize=9)
    ax_intensity.grid(axis="y", alpha=0.25)

    fig.suptitle("Climate-Scenario Extension: Extreme Rainfall Indicators", fontsize=17, y=0.98)
    fig.text(
        0.01,
        0.01,
        "Simulated ensemble for workflow demonstration; not an operational climate projection. Error bars show ensemble 10th-90th percentile range.",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.98)
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
