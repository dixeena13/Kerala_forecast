from meteo_ml.scenarios import (
    ensemble_change_summary,
    extreme_rainfall_indicators,
    make_extreme_rainfall_scenario_dataset,
)


def test_scenario_dataset_dimensions() -> None:
    ds = make_extreme_rainfall_scenario_dataset(n_years=2, n_members=3)
    assert set(ds.dims) == {"period", "member", "station", "day"}
    assert ds.sizes["period"] == 2
    assert ds.sizes["member"] == 3
    assert ds.sizes["day"] == 730


def test_extreme_indicator_outputs() -> None:
    ds = make_extreme_rainfall_scenario_dataset(n_years=2, n_members=3)
    indicators = extreme_rainfall_indicators(ds)
    assert "heavy_rain_days_per_year" in indicators
    assert "annual_max_daily_rainfall" in indicators
    assert "p95_daily_rainfall" in indicators
    assert indicators["heavy_rain_days_per_year"].sizes["station"] == ds.sizes["station"]


def test_ensemble_change_summary_contains_spread() -> None:
    ds = make_extreme_rainfall_scenario_dataset(n_years=2, n_members=3)
    summary = ensemble_change_summary(extreme_rainfall_indicators(ds))
    assert "heavy_rain_days_per_year_change_mean" in summary
    assert "heavy_rain_days_per_year_change_p10" in summary
    assert "heavy_rain_days_per_year_change_p90" in summary
