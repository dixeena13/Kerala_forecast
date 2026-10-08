from meteo_ml.ops_weather_products import HAZARDS, mission_profiles, product_matrix_records, simple_mission_risk, summarize_requirements


def test_product_matrix_covers_all_profiles_and_hazards() -> None:
    profiles = mission_profiles()
    records = product_matrix_records()
    assert len(records) == len(profiles) * len(HAZARDS)
    assert {record["hazard"] for record in records} == set(HAZARDS)


def test_requirements_have_operational_questions() -> None:
    requirements = summarize_requirements()
    assert requirements
    assert all(item["operational_question"] for item in requirements)
    assert all(item["recommended_products"] for item in requirements)


def test_simple_mission_risk_is_normalized() -> None:
    profile = mission_profiles()[0]
    score = simple_mission_risk(profile, {hazard: 1.0 for hazard in HAZARDS})
    assert score == 1.0
    score = simple_mission_risk(profile, {hazard: 0.0 for hazard in HAZARDS})
    assert score == 0.0
