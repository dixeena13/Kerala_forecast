import numpy as np
import pytest
import xarray as xr
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from meteo_ml.verification import (classify_events, make_verification_demo,
                                   verify_events, verify_rainfall)


def flags(values):
    return xr.DataArray(values, dims='time', coords={'time': np.arange(len(values))})


def test_hand_calculated_contingency():
    result = verify_events(flags([1, 1, 0, 0]), flags([1, 0, 1, 0]), ['time'])
    for name in ['hits', 'misses', 'false_alarms', 'correct_negatives']:
        assert result[name].item() == 1
    assert result.pod.item() == .5
    assert result.far.item() == .5
    assert result.csi.item() == pytest.approx(1/3)
    assert result.frequency_bias.item() == 1
    assert classify_events(flags([1, 1, 0, 0]), flags([1, 0, 1, 0])).values.tolist() == [1, 3, 2, 0]


def test_missing_and_zero_denominators():
    result = verify_events(flags([np.nan, 0, 1]), flags([1, 0, np.nan]), ['time'])
    assert result.valid_pairs == 1 and result.missing_pairs == 2
    assert result.correct_negatives == 1
    assert np.isnan(result.pod) and np.isnan(result.far) and np.isnan(result.csi)
    empty = verify_events(flags([np.nan]), flags([np.nan]), ['time'])
    assert empty.valid_pairs == 0 and np.isnan(empty.csi)


def test_perfect_and_no_warning_forecasts():
    perfect = verify_events(flags([1, 0, 1]), flags([1, 0, 1]), ['time'])
    assert perfect.pod == 1 and perfect.csi == 1 and perfect.far == 0
    absent = verify_events(flags([0, 0]), flags([1, 0]), ['time'])
    assert absent.pod == 0 and absent.csi == 0 and np.isnan(absent.far)


def test_reject_coordinate_mismatch_and_invalid_flags():
    with pytest.raises(ValueError, match='Coordinates'):
        verify_events(flags([1, 0]), flags([1, 0]).assign_coords(time=[1, 2]), ['time'])
    with pytest.raises(ValueError, match='Duplicate'):
        verify_events(flags([1, 0]).assign_coords(time=[0, 0]), flags([1, 0]), ['time'])
    with pytest.raises(ValueError, match='flags'):
        verify_events(flags([2, 0]), flags([1, 0]), ['time'])
    with pytest.raises(ValueError, match='reduce_dims'):
        verify_events(flags([1]), flags([1]), ['station'])


def test_station_grid_consistency_and_reproducibility():
    station, grid = make_verification_demo()
    assert station.identical(make_verification_demo()[0])
    for ds in [station, grid]:
        by_location = verify_rainfall(ds.forecast_rainfall, ds.reference_rainfall, 20, ['time'])
        pooled = verify_rainfall(ds.forecast_rainfall, ds.reference_rainfall, 20, list(ds.forecast_rainfall.dims))
        for name in ['hits', 'misses', 'false_alarms', 'correct_negatives', 'valid_pairs', 'missing_pairs']:
            assert by_location[name].sum() == pooled[name]
        assert pooled.valid_pairs + pooled.missing_pairs == ds.forecast_rainfall.size


def test_threshold_boundary_and_units():
    forecast = flags([20, 19.9, np.nan])
    reference = flags([20, 20, 0])
    for array in [forecast, reference]:
        array.attrs = {'units': 'mm', 'accumulation_hours': 24}
    result = verify_rainfall(forecast, reference, 20, ['time'])
    assert result.hits == 1 and result.misses == 1 and result.missing_pairs == 1
    with pytest.raises(ValueError, match='positive'):
        verify_rainfall(forecast, reference, 0, ['time'])
    forecast.attrs['units'] = 'm'
    with pytest.raises(ValueError, match='units'):
        verify_rainfall(forecast, reference, 20, ['time'])


def test_invalid_rainfall_and_accumulation():
    forecast = flags([-1, 2])
    reference = flags([0, 2])
    for array in [forecast, reference]:
        array.attrs = {'units': 'mm', 'accumulation_hours': 24}
    with pytest.raises(ValueError, match='nonnegative'):
        verify_rainfall(forecast, reference, 20, ['time'])
    forecast[:] = 1
    forecast.attrs['accumulation_hours'] = 6
    with pytest.raises(ValueError, match='accumulation'):
        verify_rainfall(forecast, reference, 20, ['time'])


def test_runner_exports_consistent_products(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'MPLCONFIGDIR': str(tmp_path / 'mpl')}
    subprocess.run([sys.executable, str(root / 'scripts/verify_rainfall_warnings.py'),
                    '--output-dir', str(tmp_path), '--threshold', '30'],
                   env=env, check=True, capture_output=True, text=True)
    products = tmp_path / 'products'
    with (products / 'warning_verification_demo_summary.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8  # Four thresholds per layout, including custom threshold.
    for row in rows:
        total = sum(int(row[k]) for k in ['hits', 'misses', 'false_alarms', 'correct_negatives'])
        assert total == int(row['valid_pairs'])
    with (products / 'warning_verification_demo_locations.csv').open() as handle:
        locations = list(csv.DictReader(handle))
    assert len(locations) == 3 + 8*6
    assert all(float(row['threshold_mm_24h']) == 30 for row in locations)
    manifest = json.loads((products / 'warning_verification_demo_manifest.json').read_text())
    assert manifest['seed'] == 23 and manifest['threshold_mm_24h'] == 30
    for layout in ['station', 'grid']:
        assert (tmp_path / 'figures' / f'warning_verification_demo_{layout}.png').stat().st_size > 10000


def test_grid_is_same_counting_problem_as_flat_samples():
    forecast = xr.DataArray([[1, 1], [0, 0]], dims=['time', 'cell'],
                            coords={'time': [0, 1], 'cell': [0, 1]})
    observed = forecast.copy(data=[[1, 0], [1, 0]])
    grid = verify_events(forecast, observed, ['time', 'cell'])
    flat = verify_events(flags(forecast.values.ravel()), flags(observed.values.ravel()), ['time'])
    for name in grid.data_vars:
        assert grid[name].item() == flat[name].item()
