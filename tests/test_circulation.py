import numpy as np
import pytest

from meteo_ml.circulation import analyse_circulation, make_circulation_demo


def test_reproducible_counts_and_prescribed_association():
    ds = make_circulation_demo()
    records, annual = analyse_circulation(ds)
    assert ds.identical(make_circulation_demo())
    assert len(records) == 9 and len(annual) == 90
    for station in ds.station.values:
        rows = [r for r in records if r['station'] == station]
        assert sum(r['valid_days'] for r in rows) == ds.sizes['time']
        assert rows[2]['frequency_percent'] > rows[0]['frequency_percent']


def test_threshold_does_not_use_later_rainfall():
    ds = make_circulation_demo()
    original, _ = analyse_circulation(ds)
    ds.precipitation.loc[dict(time=slice('2006', None))] *= 10
    changed, _ = analyse_circulation(ds)
    assert [r['threshold_mm'] for r in original] == [r['threshold_mm'] for r in changed]


def test_missing_values_are_not_dry_days():
    ds = make_circulation_demo()
    ds.precipitation.values[:10, 0] = np.nan
    rows, _ = analyse_circulation(ds)
    assert sum(r['valid_days'] for r in rows if r['station'] == 'Kochi') == ds.sizes['time'] - 10


def test_invalid_inputs_fail():
    ds = make_circulation_demo()
    with pytest.raises(ValueError, match='daily'):
        analyse_circulation(ds.isel(time=slice(None, None, 2)))
    ds.circulation_index[:] = 0
    with pytest.raises(ValueError, match='nonconstant'):
        analyse_circulation(ds)
