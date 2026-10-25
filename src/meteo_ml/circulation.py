"""Descriptive rainfall extremes conditional on a daily circulation index."""
from __future__ import annotations

import numpy as np
import xarray as xr


def make_circulation_demo(seed: int = 17) -> xr.Dataset:
    """Generate a prescribed association, not a reconstruction of real weather."""
    rng = np.random.default_rng(seed)
    time = np.arange(np.datetime64('1991-01-01'), np.datetime64('2021-01-01'))
    index = np.zeros(len(time))
    for i in range(1, len(time)):
        index[i] = 0.7 * index[i - 1] + rng.normal(0, 0.7)
    month = xr.DataArray(time, dims='time').dt.month.values
    monsoon = (month >= 6) & (month <= 9)
    wet = rng.random((len(time), 3)) < (0.2 + 0.4 * monsoon[:, None])
    rain = wet * rng.gamma(1.4, 8, (len(time), 3))
    # Positive index values intentionally amplify rainfall in this toy example.
    rain *= np.exp(0.3 * index[:, None]) * np.array([1, 1.6, 1.4])
    return xr.Dataset(
        {'precipitation': (('time', 'station'), rain, {'units': 'mm/day'}),
         'circulation_index': ('time', index, {'units': '1'})},
        coords={'time': time, 'station': ['Kochi', 'Idukki', 'Wayanad']},
        attrs={'source': 'Simulated daily rainfall and persistent toy circulation index; seed=17.'},
    )


def analyse_circulation(ds: xr.Dataset, baseline_start: str = '1991-01-01',
                        baseline_end: str = '2005-12-31') -> tuple[list[dict], list[dict]]:
    """Use reference wet-day p95 and reference-standardized index terciles.

    Frequencies use only paired valid days. No causal or significance inference
    is made; index bins are not diagnosed physical circulation regimes.
    """
    rain, index = ds['precipitation'], ds['circulation_index']
    if rain.dims != ('time', 'station') or index.dims != ('time',):
        raise ValueError('Expected precipitation(time, station) and circulation_index(time).')
    if rain.attrs.get('units') != 'mm/day':
        raise ValueError('Convert precipitation to daily amounts and set units="mm/day".')
    times = ds.time.values
    if len(times) < 2 or not np.all(np.diff(times) == np.timedelta64(1, 'D')):
        raise ValueError('Time must be unique, increasing and daily with no missing dates.')
    if bool((rain < 0).any()):
        raise ValueError('Negative precipitation is invalid.')
    ref = ds.sel(time=slice(baseline_start, baseline_end))
    if ref.sizes['time'] < 365:
        raise ValueError('Reference period must contain at least 365 daily records.')
    ref_index = ref.circulation_index.values
    finite = ref_index[np.isfinite(ref_index)]
    if len(finite) < 100 or np.std(finite) == 0:
        raise ValueError('Reference index needs at least 100 valid, nonconstant values.')
    z = (index.values - finite.mean()) / finite.std()
    edges = np.quantile((finite - finite.mean()) / finite.std(), [1/3, 2/3])
    if edges[0] == edges[1]:
        raise ValueError('Reference index terciles are degenerate.')
    bins = np.digitize(z, edges)
    years = ds.time.dt.year.values
    records, annual = [], []
    for station in ds.station.values:
        values = rain.sel(station=station).values
        ref_rain = ref.precipitation.sel(station=station).values
        wet = ref_rain[np.isfinite(ref_rain) & (ref_rain >= 1)]
        if len(wet) < 100:
            raise ValueError(f'{station}: reference period needs at least 100 wet days.')
        threshold = float(np.quantile(wet, .95))
        valid = np.isfinite(values) & np.isfinite(z)
        event = values > threshold
        for group, label in enumerate(['Lower third', 'Middle third', 'Upper third']):
            mask = valid & (bins == group)
            n, k = int(mask.sum()), int((mask & event).sum())
            records.append(dict(station=str(station), index_bin=label, valid_days=n,
                                extreme_days=k, frequency_percent=100*k/n if n else float('nan'),
                                threshold_mm=threshold))
        for year in np.unique(years):
            mask = valid & (years == year)
            n, k = int(mask.sum()), int((mask & event).sum())
            annual.append(dict(station=str(station), year=int(year), valid_days=n,
                               extreme_days=k, frequency_percent=100*k/n if n else float('nan')))
    return records, annual
