from __future__ import annotations

from dataclasses import dataclass


HAZARDS = ["wind", "precipitation", "visibility", "thunderstorm", "freezing_level", "heat_stress"]


@dataclass(frozen=True)
class MissionProfile:
    name: str
    user_group: str
    operational_question: str
    hazard_weights: dict[str, int]
    recommended_products: tuple[str, ...]
    update_frequency: str


MISSION_PROFILES: tuple[MissionProfile, ...] = (
    MissionProfile(
        name="Air operations",
        user_group="aviation coordination",
        operational_question="Can aircraft or drones operate safely in the next 6-24 hours?",
        hazard_weights={"wind": 5, "precipitation": 3, "visibility": 5, "thunderstorm": 5, "freezing_level": 3, "heat_stress": 1},
        recommended_products=("wind gust nowcast", "visibility forecast", "thunderstorm alert", "freezing-level forecast"),
        update_frequency="hourly / event-driven",
    ),
    MissionProfile(
        name="Ground mobility",
        user_group="field logistics",
        operational_question="Where can rain, snow, or poor visibility disrupt movement and supply routes?",
        hazard_weights={"wind": 2, "precipitation": 5, "visibility": 4, "thunderstorm": 3, "freezing_level": 4, "heat_stress": 2},
        recommended_products=("precipitation accumulation map", "road-weather risk layer", "visibility forecast"),
        update_frequency="3-hourly",
    ),
    MissionProfile(
        name="Mountain support",
        user_group="alpine operations",
        operational_question="Which weather hazards affect exposed terrain, freezing level, and local accessibility?",
        hazard_weights={"wind": 4, "precipitation": 4, "visibility": 4, "thunderstorm": 4, "freezing_level": 5, "heat_stress": 1},
        recommended_products=("freezing-level forecast", "wind gust forecast", "convective risk briefing", "snow/rain transition layer"),
        update_frequency="3-hourly / briefing before deployment",
    ),
    MissionProfile(
        name="Civil protection support",
        user_group="coordination staff",
        operational_question="Which weather developments require early coordination with civil partners?",
        hazard_weights={"wind": 4, "precipitation": 5, "visibility": 2, "thunderstorm": 5, "freezing_level": 3, "heat_stress": 4},
        recommended_products=("impact-based warning dashboard", "heavy precipitation scenario", "thunderstorm alert", "heat stress outlook"),
        update_frequency="daily / event-driven",
    ),
)


def mission_profiles() -> list[MissionProfile]:
    return list(MISSION_PROFILES)


def product_matrix_records() -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for profile in MISSION_PROFILES:
        for hazard in HAZARDS:
            records.append(
                {
                    "mission_profile": profile.name,
                    "user_group": profile.user_group,
                    "hazard": hazard,
                    "importance": profile.hazard_weights[hazard],
                    "recommended_products": "; ".join(profile.recommended_products),
                    "update_frequency": profile.update_frequency,
                    "operational_question": profile.operational_question,
                }
            )
    return records


def summarize_requirements() -> list[dict[str, str]]:
    return [
        {
            "mission_profile": profile.name,
            "user_group": profile.user_group,
            "operational_question": profile.operational_question,
            "recommended_products": "; ".join(profile.recommended_products),
            "update_frequency": profile.update_frequency,
        }
        for profile in MISSION_PROFILES
    ]


def simple_mission_risk(profile: MissionProfile, hazard_levels: dict[str, float]) -> float:
    """Calculate a weighted risk score from normalized hazard levels.

    Hazard levels are expected on a 0-1 scale. The score is also returned on a
    0-1 scale and is intended for product-design demonstrations, not operations.
    """
    numerator = 0.0
    denominator = 0.0
    for hazard, weight in profile.hazard_weights.items():
        numerator += weight * min(max(hazard_levels.get(hazard, 0.0), 0.0), 1.0)
        denominator += weight
    return round(numerator / denominator, 3) if denominator else 0.0
