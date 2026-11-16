"""Exact-match binary event verification for stations and regular grids."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr


def _matched(forecast: xr.DataArray, observed: xr.DataArray) -> None:
    if forecast.dims != observed.dims or forecast.sizes != observed.sizes:
        raise ValueError('Forecast and reference must have identical dimensions and sizes.')
    for dim in forecast.dims:
        if dim not in forecast.coords or dim not in observed.coords:
            raise ValueError(f'Explicit coordinates required for {dim}.')
        if not forecast.get_index(dim).is_unique or not observed.get_index(dim).is_unique:
            raise ValueError(f'Duplicate coordinate labels in {dim}.')
    try:
        xr.align(forecast, observed, join='exact')
    except ValueError as exc:
        raise ValueError('Coordinates must match exactly; align inputs explicitly first.') from exc
    for name in set(forecast.coords) | set(observed.coords):
        if name not in forecast.coords or name not in observed.coords:
            raise ValueError(f'Coordinate {name} must be present on both inputs.')
        if not forecast.coords[name].equals(observed.coords[name]):
            raise ValueError(f'Coordinate {name} differs between inputs.')


def classify_events(forecast: xr.DataArray, observed: xr.DataArray) -> xr.DataArray:
    """Return codes: 0 correct negative, 1 hit, 2 miss, 3 false alarm; NaN invalid.

    Inputs are binary flags (0/1 or bool). Only NaN may represent missing data.
    """
    _matched(forecast, observed)
    for array in (forecast, observed):
        if bool((~(array.isin([0, 1]) | array.isnull())).any()):
            raise ValueError('Event flags must be 0, 1 or NaN.')
    valid = forecast.notnull() & observed.notnull()
    codes = xr.where(observed == 1, xr.where(forecast == 1, 1, 2),
                     xr.where(forecast == 1, 3, 0)).where(valid)
    return codes.rename('outcome')


def verify_events(forecast: xr.DataArray, observed: xr.DataArray,
                  reduce_dims: Sequence[str]) -> xr.Dataset:
    """Pool contingency counts before computing scores; zero denominators -> NaN."""
    dims = list(reduce_dims)
    if not dims or len(set(dims)) != len(dims) or any(d not in forecast.dims for d in dims):
        raise ValueError('reduce_dims must contain distinct existing dimensions.')
    outcome = classify_events(forecast, observed)
    result = xr.Dataset({name: (outcome == code).sum(dims) for name, code in
                         [('correct_negatives', 0), ('hits', 1), ('misses', 2), ('false_alarms', 3)]})
    result['valid_pairs'] = outcome.notnull().sum(dims)
    result['missing_pairs'] = outcome.isnull().sum(dims)
    h, m, f = result.hits, result.misses, result.false_alarms
    for name, numerator, denominator in [
        ('pod', h, h + m), ('far', f, h + f), ('csi', h, h + m + f),
        ('frequency_bias', h + f, h + m),
    ]:
        result[name] = numerator / denominator.where(denominator > 0)
    result.attrs['method'] = 'Exact paired sample verification; unweighted pooled counts.'
    return result


def rainfall_events(forecast: xr.DataArray, observed: xr.DataArray,
                    threshold_mm: float) -> tuple[xr.DataArray, xr.DataArray]:
    """Threshold already matched 24-hour rainfall amounts, inclusive at threshold."""
    _matched(forecast, observed)
    if not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError('Rainfall threshold must be finite and positive.')
    for array in (forecast, observed):
        if array.attrs.get('units') != 'mm':
            raise ValueError('Daily accumulation units must be mm; convert inputs explicitly.')
        if array.attrs.get('accumulation_hours') != 24:
            raise ValueError('Set accumulation_hours=24 for matched daily amounts.')
        if bool(((array < 0) | np.isinf(array)).any()):
            raise ValueError('Rainfall must be nonnegative and finite, or NaN.')
    return tuple((array >= threshold_mm).where(array.notnull()) for array in (forecast, observed))


def verify_rainfall(forecast: xr.DataArray, observed: xr.DataArray,
                    threshold_mm: float, reduce_dims: Sequence[str]) -> xr.Dataset:
    result = verify_events(*rainfall_events(forecast, observed, threshold_mm), reduce_dims)
    result.attrs.update(threshold_mm=float(threshold_mm), accumulation_hours=24,
                        event_definition='rainfall >= threshold; not an official warning level')
    return result


def make_verification_demo(seed: int = 23) -> tuple[xr.Dataset, xr.Dataset]:
    """Create paired toy rainfall at stations and a separate grid, not observations."""
    rng = np.random.default_rng(seed)
    time = np.arange(np.datetime64('2020-06-01'), np.datetime64('2020-09-29'))
    attrs = {'units': 'mm', 'accumulation_hours': 24}
    datasets = []
    for dims, coords, shape in [
        (('time', 'station'), {'time': time, 'station': ['Kochi', 'Idukki', 'Wayanad']}, (len(time), 3)),
        (('time', 'latitude', 'longitude'),
         {'time': time, 'latitude': np.linspace(8.5, 12.5, 8),
          'longitude': np.linspace(75.5, 77.5, 6)}, (len(time), 8, 6)),
    ]:
        reference = rng.gamma(1.2, 14, shape) * (rng.random(shape) < .55)
        proposal = np.maximum(0, reference * .85 + rng.normal(0, 9, shape))
        reference[rng.random(shape) < .025] = np.nan
        proposal[rng.random(shape) < .015] = np.nan
        datasets.append(xr.Dataset(
            {'forecast_rainfall': (dims, proposal, attrs.copy()),
             'reference_rainfall': (dims, reference, attrs.copy())},
            coords=coords, attrs={'source': 'Simulated paired daily rainfall; seed=23; not real forecast skill.'}))
    return tuple(datasets)
