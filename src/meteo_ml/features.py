from __future__ import annotations

import numpy as np
import xarray as xr


FEATURE_NAMES = [
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
]


def to_feature_matrix(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    missing = [name for name in FEATURE_NAMES + ["precipitation"] if name not in ds]
    if missing:
        raise ValueError(f"Dataset is missing required variables: {', '.join(missing)}")

    x = np.column_stack([ds[name].values for name in FEATURE_NAMES]).astype("float32")
    y = ds["precipitation"].values.astype("float32").reshape(-1, 1)
    return x, y


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return (x - mean) / std, mean, std


def apply_standardization(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.where(std == 0.0, 1.0, std)
