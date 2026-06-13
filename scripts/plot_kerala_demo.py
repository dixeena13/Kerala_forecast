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

from meteo_ml.data import make_kerala_demo_dataset


def main() -> None:
    output = Path("figures/kerala_demo_rainfall.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    ds = make_kerala_demo_dataset(n_days=90, seed=11)
    stations = ["Kochi", "Idukki", "Wayanad", "Kozhikode"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    ax_ts, ax_bar = axes

    for station in stations:
        mask = ds["station"].values == station
        day = ds["day_of_year"].values[mask]
        rain = ds["precipitation"].values[mask]
        ax_ts.plot(day, rain, linewidth=1.8, label=station)

    ax_ts.axhline(20.0, color="crimson", linestyle="--", linewidth=1.2, label="heavy rain threshold")
    ax_ts.set_title("Kerala demo rainfall time series")
    ax_ts.set_ylabel("Rainfall (mm/day)")
    ax_ts.legend(ncol=3, fontsize=9)
    ax_ts.grid(alpha=0.25)

    station_means = []
    for station in stations:
        mask = ds["station"].values == station
        station_means.append(float(np.mean(ds["precipitation"].values[mask])))

    ax_bar.bar(stations, station_means, color=["#2a9d8f", "#457b9d", "#8ab17d", "#e9c46a"])
    ax_bar.set_title("Mean rainfall by Kerala station in demo data")
    ax_bar.set_ylabel("Mean rainfall (mm/day)")
    ax_bar.set_xlabel("Station")
    ax_bar.grid(axis="y", alpha=0.25)

    fig.suptitle("Synthetic Kerala monsoon-like rainfall used for reproducible ML pipeline", fontsize=13)
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
