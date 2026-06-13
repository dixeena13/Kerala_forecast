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
    colors = {
        "Kochi": "#277da1",
        "Idukki": "#f8961e",
        "Wayanad": "#43aa8b",
        "Kozhikode": "#d62828",
    }

    fig, (ax_ts, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(11, 7.5),
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.42},
    )

    for station in stations:
        mask = ds["station"].values == station
        day = ds["day_of_year"].values[mask]
        rain = ds["precipitation"].values[mask]
        ax_ts.plot(day, rain, linewidth=1.8, color=colors[station], label=station)

    ax_ts.axhline(20.0, color="#6a040f", linestyle="--", linewidth=1.3, label="heavy rain threshold")
    ax_ts.set_title("Daily rainfall by Kerala station", fontsize=15, pad=10)
    ax_ts.set_ylabel("Rainfall (mm/day)")
    ax_ts.set_xlabel("Day of year")
    ax_ts.set_xlim(150, 245)
    ax_ts.set_ylim(bottom=0)
    ax_ts.legend(ncol=3, fontsize=9, frameon=True, loc="upper left")
    ax_ts.grid(alpha=0.25)

    station_means = []
    for station in stations:
        mask = ds["station"].values == station
        station_means.append(float(np.mean(ds["precipitation"].values[mask])))

    x_positions = np.arange(len(stations))
    ax_bar.bar(
        x_positions,
        station_means,
        width=0.58,
        color=[colors[station] for station in stations],
        edgecolor="#222222",
        linewidth=0.5,
    )
    ax_bar.set_title("Mean rainfall in demo period", fontsize=14, pad=8)
    ax_bar.set_ylabel("Mean rainfall\n(mm/day)")
    ax_bar.set_xlabel("Station")
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(stations, rotation=0, ha="center")
    ax_bar.set_ylim(0, max(station_means) * 1.25)
    ax_bar.grid(axis="y", alpha=0.25)

    for x, value in zip(x_positions, station_means):
        ax_bar.text(x, value + 0.25, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Kerala Monsoon-Like Rainfall Demo Dataset",
        fontsize=18,
        y=0.98,
    )
    fig.text(
        0.01,
        0.01,
        "Synthetic data for reproducible ML pipeline; replace with real station/radar/NWP data for operational use.",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.98)
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
