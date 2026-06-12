from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, heavy_rain_threshold_mm: float = 20.0) -> dict[str, float]:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    observed_heavy = y_true >= heavy_rain_threshold_mm
    predicted_heavy = y_pred >= heavy_rain_threshold_mm
    hits = float(np.logical_and(observed_heavy, predicted_heavy).sum())
    misses = float(np.logical_and(observed_heavy, ~predicted_heavy).sum())
    false_alarms = float(np.logical_and(~observed_heavy, predicted_heavy).sum())

    recall = hits / (hits + misses) if hits + misses > 0 else 0.0
    precision = hits / (hits + false_alarms) if hits + false_alarms > 0 else 0.0
    return {
        "mae_mm": mae,
        "rmse_mm": rmse,
        "heavy_rain_recall": recall,
        "heavy_rain_precision": precision,
    }
