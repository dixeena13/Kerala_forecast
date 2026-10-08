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

from meteo_ml.ops_weather_products import HAZARDS, mission_profiles


def main() -> None:
    output = Path("figures/ops_weather_product_decision_matrix.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    profiles = mission_profiles()
    matrix = np.array([[profile.hazard_weights[hazard] for hazard in HAZARDS] for profile in profiles], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=1, vmax=5)

    ax.set_xticks(np.arange(len(HAZARDS)))
    ax.set_yticks(np.arange(len(profiles)))
    ax.set_xticklabels([hazard.replace("_", " ").title() for hazard in HAZARDS], rotation=35, ha="right")
    ax.set_yticklabels([profile.name for profile in profiles])

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{int(matrix[row, col])}", ha="center", va="center", color="#111827", fontsize=10, weight="bold")

    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
    colorbar.set_label("Requirement importance (1 low - 5 high)")

    ax.set_title("Weather Product Requirement Matrix for Operational Use Cases", fontsize=15, pad=12)
    ax.set_xlabel("Weather hazard / product driver")
    ax.set_ylabel("Mission profile")
    ax.text(
        0.0,
        -0.26,
        "Simulated business-analysis artifact: maps user mission profiles to meteorological product priorities.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5563",
    )
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
