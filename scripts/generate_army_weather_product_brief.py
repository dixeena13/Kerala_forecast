from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meteo_ml.army_products import product_matrix_records, summarize_requirements


def write_csv(records: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_markdown(requirements: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Army Weather Product Brief",
        "",
        "This is a simulated product-design brief for portfolio purposes, not an operational MeteoSwiss or Swiss Army product.",
        "It demonstrates how user requirements can be translated into weather-product concepts and update frequencies.",
        "",
        "| Mission profile | User group | Operational question | Recommended weather products | Update frequency |",
        "|---|---|---|---|---|",
    ]
    for item in requirements:
        lines.append(
            "| {mission_profile} | {user_group} | {operational_question} | {recommended_products} | {update_frequency} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Product Management Notes",
            "",
            "- Start with mission-specific questions rather than a generic weather dashboard.",
            "- Translate each question into hazards, thresholds, product formats, and update frequency.",
            "- Keep traceability from user requirement to meteorological variable and IT product component.",
            "- Review products with users after exercises or high-impact weather events.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    write_csv(product_matrix_records(), Path("products/army_weather_product_matrix.csv"))
    write_markdown(summarize_requirements(), Path("products/army_weather_product_brief.md"))
    print("saved products/army_weather_product_matrix.csv")
    print("saved products/army_weather_product_brief.md")


if __name__ == "__main__":
    main()
