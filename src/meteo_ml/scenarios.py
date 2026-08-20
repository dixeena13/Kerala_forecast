from __future__ import annotations

import numpy as np
import xarray as xr

from meteo_ml.kerala import KERALA_STATIONS


def make_extreme_rainfall_scenario_dataset(
    n_years: int = 30,
    n_members: int = 8,
    seed: int = 42,
) -> xr.Dataset:
    """Create a small simulated ensemble for climate-scenario indicator demos.

    The dataset mimics a baseline and future climate-scenario product workflow:
    multiple ensemble members, station-level daily rainfall, and a future signal
    with more intense heavy-rain days. It is simulated demo data, not a climate
    projection product.
    """
    rng = np.random.default_rng(seed)
    periods = np.array(["baseline", "future"], dtype=object)
    members = np.array([f"member_{idx + 1:02d}" for idx in range(n_members)], dtype=object)
    stations = np.array([station.name for station in KERALA_STATIONS], dtype=object)
    days = np.arange(n_years * 365)

    rainfall = np.zeros((len(periods), n_members, len(stations), len(days)), dtype="float32")

    for period_idx, period in enumerate(periods):
        future_multiplier = 1.0 if period == "baseline" else 1.18
        future_heavy_boost = 0.0 if period == "baseline" else 0.08

        for member_idx in range(n_members):
            member_factor = rng.normal(1.0, 0.08)
            for station_idx, station in enumerate(KERALA_STATIONS):
                orographic_factor = 1.0 + station.elevation_m / 1800.0
                coastal_factor = 1.0 + max(0.0, 1.0 - station.coast_distance_km / 90.0) * 0.25
                monsoon_cycle = 0.5 + 0.5 * np.sin(2.0 * np.pi * ((days % 365) - 150) / 365.0)
                wet_probability = np.clip(0.18 + 0.42 * monsoon_cycle + future_heavy_boost, 0.03, 0.85)
                wet_days = rng.random(len(days)) < wet_probability

                base_amount = rng.gamma(shape=1.3, scale=7.0, size=len(days))
                extreme_component = rng.gamma(shape=1.1, scale=24.0, size=len(days))
                extreme_days = rng.random(len(days)) < (0.035 + future_heavy_boost / 2.0)

                amount = wet_days * base_amount + wet_days * extreme_days * extreme_component
                amount *= future_multiplier * member_factor * orographic_factor * coastal_factor
                rainfall[period_idx, member_idx, station_idx, :] = amount.astype("float32")

    return xr.Dataset(
        {
            "precipitation": (("period", "member", "station", "day"), rainfall),
        },
        coords={
            "period": periods,
            "member": members,
            "station": stations,
            "day": days,
        },
        attrs={
            "title": "Simulated extreme-rainfall climate scenario ensemble",
            "region": "Kerala, India / South Asia",
            "data_note": "Simulated portfolio data for scenario-method demonstration; not an operational climate projection.",
        },
    )


def extreme_rainfall_indicators(ds: xr.Dataset, heavy_threshold_mm: float = 20.0) -> xr.Dataset:
    """Calculate station/member extreme-rainfall indicators from daily precipitation."""
    precipitation = ds["precipitation"]
    heavy_rain_days_per_year = (precipitation >= heavy_threshold_mm).sum("day") / (precipitation.sizes["day"] / 365.0)
    annual_max_daily_rainfall = precipitation.coarsen(day=365, boundary="trim").max().mean("day")
    p95_daily_rainfall = precipitation.quantile(0.95, dim="day")

    return xr.Dataset(
        {
            "heavy_rain_days_per_year": heavy_rain_days_per_year,
            "annual_max_daily_rainfall": annual_max_daily_rainfall,
            "p95_daily_rainfall": p95_daily_rainfall,
        },
        attrs={
            "heavy_threshold_mm": heavy_threshold_mm,
            "note": "Indicators are calculated from simulated ensemble data for workflow demonstration.",
        },
    )


def _drop_quantile_coord(array: xr.DataArray) -> xr.DataArray:
    if "quantile" in array.coords:
        array = array.drop_vars("quantile")
    return array


def ensemble_change_summary(indicators: xr.Dataset) -> xr.Dataset:
    """Return future-minus-baseline change summaries across ensemble members."""
    future = indicators.sel(period="future")
    baseline = indicators.sel(period="baseline")
    change = future - baseline

    variables = {}
    for name, variable in change.data_vars.items():
        variables[f"{name}_change_mean"] = variable.mean("member")
        variables[f"{name}_change_p10"] = _drop_quantile_coord(variable.quantile(0.10, dim="member"))
        variables[f"{name}_change_p90"] = _drop_quantile_coord(variable.quantile(0.90, dim="member"))
    return xr.Dataset(variables)



def _risk_message(heavy_days_change: float, annual_max_change: float) -> str:
    if heavy_days_change >= 25.0 or annual_max_change >= 60.0:
        return "large increase in extreme-rainfall indicators"
    if heavy_days_change >= 15.0 or annual_max_change >= 35.0:
        return "moderate increase in extreme-rainfall indicators"
    if heavy_days_change > 0.0 or annual_max_change > 0.0:
        return "small increase in extreme-rainfall indicators"
    return "no increase in this simulated ensemble summary"


def scenario_product_records(summary: xr.Dataset) -> list[dict[str, str | float]]:
    """Convert scenario change indicators into user-facing records."""
    records: list[dict[str, str | float]] = []
    for station in summary["station"].values:
        heavy_mean = float(summary["heavy_rain_days_per_year_change_mean"].sel(station=station))
        heavy_p10 = float(summary["heavy_rain_days_per_year_change_p10"].sel(station=station))
        heavy_p90 = float(summary["heavy_rain_days_per_year_change_p90"].sel(station=station))
        annual_max_mean = float(summary["annual_max_daily_rainfall_change_mean"].sel(station=station))
        p95_mean = float(summary["p95_daily_rainfall_change_mean"].sel(station=station))
        records.append(
            {
                "station": str(station),
                "heavy_rain_days_change_mean": round(heavy_mean, 2),
                "heavy_rain_days_change_p10": round(heavy_p10, 2),
                "heavy_rain_days_change_p90": round(heavy_p90, 2),
                "annual_max_daily_rainfall_change_mm": round(annual_max_mean, 2),
                "p95_daily_rainfall_change_mm": round(p95_mean, 2),
                "interpretation": _risk_message(heavy_mean, annual_max_mean),
            }
        )
    return records
