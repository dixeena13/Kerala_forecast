from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def make_synthetic_dataset(n_samples: int = 256, seed: int = 7) -> xr.Dataset:
    """Create a small weather-like dataset for reproducible smoke tests."""
    rng = np.random.default_rng(seed)
    temperature = rng.normal(294.0, 5.0, n_samples)
    humidity = np.clip(rng.normal(0.72, 0.16, n_samples), 0.0, 1.0)
    pressure = rng.normal(1008.0, 8.0, n_samples)
    wind_u = rng.normal(0.0, 4.0, n_samples)
    wind_v = rng.normal(0.0, 4.0, n_samples)

    instability = np.maximum(temperature - 292.0, 0.0) * humidity
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
    precipitation = np.maximum(
        0.0,
        0.55 * instability + 0.08 * wind_speed - 0.025 * (pressure - 1008.0)
        + rng.normal(0.0, 0.35, n_samples),
    )

    return xr.Dataset(
        {
            "temperature": ("sample", temperature),
            "humidity": ("sample", humidity),
            "pressure": ("sample", pressure),
            "wind_u": ("sample", wind_u),
            "wind_v": ("sample", wind_v),
            "precipitation": ("sample", precipitation),
        },
        coords={"sample": np.arange(n_samples)},
        attrs={"description": "Synthetic data shaped like a local precipitation postprocessing table."},
    )


def load_dataset(path: str | Path | None = None) -> xr.Dataset:
    """Load a netCDF dataset, or return synthetic data when no path is supplied.

    GRIB support can be added by installing the optional `grib` dependencies and
    calling `xr.open_dataset(path, engine="cfgrib")`.
    """
    if path is None:
        return make_synthetic_dataset()
    return xr.open_dataset(path)

