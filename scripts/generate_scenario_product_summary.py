from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/meteo-ml-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meteo_ml.scenarios import (
    ensemble_change_summary,
    extreme_rainfall_indicators,
    make_extreme_rainfall_scenario_dataset,
    scenario_product_records,
)


def write_csv(records: list[dict[str, str | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_markdown(records: list[dict[str, str | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Extreme Rainfall Scenario Product Summary",
        "",
        "This is a simulated portfolio product summary, not an operational climate projection.",
        "It demonstrates how ensemble climate information can be translated into station-level indicators for users.",
        "",
        "| Station | Heavy-rain days change (days/year) | Ensemble range | Annual max change (mm/day) | 95th percentile change (mm/day) | Interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| {station} | {heavy:.2f} | {p10:.2f} to {p90:.2f} | {annual:.2f} | {p95:.2f} | {interpretation} |".format(
                station=record["station"],
                heavy=record["heavy_rain_days_change_mean"],
                p10=record["heavy_rain_days_change_p10"],
                p90=record["heavy_rain_days_change_p90"],
                annual=record["annual_max_daily_rainfall_change_mm"],
                p95=record["p95_daily_rainfall_change_mm"],
                interpretation=record["interpretation"],
            )
        )
    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "Positive values indicate a simulated future increase compared with the baseline ensemble.",
            "The ensemble range summarizes the 10th-90th percentile spread across members.",
            "For an operational product, the simulated ensemble would be replaced by climate-model projections and documented assumptions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ds = make_extreme_rainfall_scenario_dataset(n_years=30, n_members=10, seed=18)
    indicators = extreme_rainfall_indicators(ds)
    summary = ensemble_change_summary(indicators)
    records = scenario_product_records(summary)

    write_csv(records, Path("products/extreme_rainfall_scenario_summary.csv"))
    write_markdown(records, Path("products/extreme_rainfall_scenario_summary.md"))
    print("saved products/extreme_rainfall_scenario_summary.csv")
    print("saved products/extreme_rainfall_scenario_summary.md")


if __name__ == "__main__":
    main()
