from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from meteo_ml.kerala import KERALA_STATIONS


def make_kerala_demo_dataset(n_days: int = 180, seed: int = 7) -> xr.Dataset:
    """Create Kerala monsoon-like station data for a reproducible prototype.

    The data are synthetic, but the variables are shaped like fields commonly used
    for local rainfall postprocessing: NWP precipitation, humidity, pressure, wind,
    temperature, topography, coastal exposure, and day of year.
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | str]] = []

    for day in range(n_days):
        day_of_year = 152 + day  # roughly June-November monsoon/post-monsoon period
        monsoon_phase = np.sin(2.0 * np.pi * (day / max(n_days - 1, 1)))
        active_spell = rng.gamma(shape=1.6, scale=5.5) if rng.random() < 0.42 else rng.gamma(0.8, 1.4)

        for station in KERALA_STATIONS:
            orographic_boost = station.elevation_m / 900.0
            coastal_boost = max(0.0, 1.0 - station.coast_distance_km / 90.0)
            latitude_gradient = (station.latitude - 8.5) / 3.5

            humidity = np.clip(0.76 + 0.12 * monsoon_phase + 0.04 * coastal_boost + rng.normal(0.0, 0.05), 0.45, 0.99)
            temperature = 299.0 - 3.2 * orographic_boost - 0.7 * monsoon_phase + rng.normal(0.0, 1.0)
            pressure = 1007.0 - 1.8 * monsoon_phase - 0.7 * active_spell / 10.0 + rng.normal(0.0, 1.2)
            wind_u = -4.0 - 1.8 * monsoon_phase + rng.normal(0.0, 1.5)
            wind_v = 1.2 + 0.6 * latitude_gradient + rng.normal(0.0, 1.1)
            nwp_precipitation = max(0.0, active_spell * (0.8 + 0.45 * coastal_boost + 0.35 * orographic_boost) + rng.normal(0.0, 2.0))

            observed = max(
                0.0,
                0.62 * nwp_precipitation
                + 22.0 * max(humidity - 0.72, 0.0)
                + 0.012 * station.elevation_m
                + 1.8 * coastal_boost
                - 0.22 * (pressure - 1005.0)
                + rng.normal(0.0, 2.4),
            )

            records.append(
                {
                    "station": station.name,
                    "district": station.district,
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "elevation_m": station.elevation_m,
                    "coast_distance_km": station.coast_distance_km,
                    "day_of_year": float(day_of_year),
                    "temperature": float(temperature),
                    "humidity": float(humidity),
                    "pressure": float(pressure),
                    "wind_u": float(wind_u),
                    "wind_v": float(wind_v),
                    "nwp_precipitation": float(nwp_precipitation),
                    "precipitation": float(observed),
                }
            )

    sample = np.arange(len(records))
    numeric_names = [
        "latitude",
        "longitude",
        "elevation_m",
        "coast_distance_km",
        "day_of_year",
        "temperature",
        "humidity",
        "pressure",
        "wind_u",
        "wind_v",
        "nwp_precipitation",
        "precipitation",
    ]
    data_vars = {name: ("sample", np.array([row[name] for row in records], dtype="float32")) for name in numeric_names}
    data_vars["station"] = ("sample", np.array([row["station"] for row in records], dtype=object))
    data_vars["district"] = ("sample", np.array([row["district"] for row in records], dtype=object))

    return xr.Dataset(
        data_vars,
        coords={"sample": sample},
        attrs={
            "title": "Kerala rainfall postprocessing demo dataset",
            "region": "Kerala, India / South Asia",
            "data_note": "Synthetic demo data for portfolio use; replace with real NWP and station/radar observations for operations.",
        },
    )


def make_synthetic_dataset(n_samples: int = 256, seed: int = 7) -> xr.Dataset:
    n_days = max(1, int(np.ceil(n_samples / len(KERALA_STATIONS))))
    return make_kerala_demo_dataset(n_days=n_days, seed=seed).isel(sample=slice(0, n_samples))


def load_dataset(path: str | Path | None = None) -> xr.Dataset:
    """Load a netCDF dataset, or return Kerala demo data when no path is supplied.

    GRIB support can be added by installing the optional `grib` dependencies and
    calling `xr.open_dataset(path, engine="cfgrib")`.
    """
    if path is None:
        return make_kerala_demo_dataset()
    return xr.open_dataset(path)
