import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from meteo_ml.solar_energy import (Economics, SolarSite, annual_summary, energy_balance,
                                   make_solar_demo, model_pv, sensitivity_cases)


def test_hourly_energy_conservation():
    time = pd.date_range('2023-01-01', periods=3, freq='h')
    pv = xr.DataArray([0, 10, 30], dims='time', coords={'time': time})
    load = pv.copy(data=[5, 20, 20])
    balance = energy_balance(pv, load)
    np.testing.assert_allclose(balance.self_consumed_kwh, [0, 10, 20])
    xr.testing.assert_equal(balance.pv_kwh, (balance.self_consumed_kwh + balance.export_kwh).rename('pv_kwh'))
    np.testing.assert_allclose(balance.demand_kwh, balance.self_consumed_kwh + balance.import_kwh)


def test_bad_inputs_and_partial_year():
    with pytest.raises(ValueError):
        SolarSite(capacity_kwp=-1)
    with pytest.raises(ValueError):
        Economics(import_inr_per_kwh=-1)
    time = pd.date_range('2023-01-01', periods=3, freq='h')
    pv = xr.DataArray([0, 1, 2], dims='time', coords={'time': time})
    with pytest.raises(ValueError):
        energy_balance(pv, pv.assign_coords(time=time+pd.Timedelta(hours=1)))
    with pytest.raises(ValueError, match='finite'):
        energy_balance(pv, pv.copy(data=[0, np.nan, 1]))
    with pytest.raises(ValueError, match='complete'):
        annual_summary(energy_balance(pv, pv), SolarSite(), Economics())


def test_financial_calculation_and_no_payback():
    time = pd.date_range('2023-01-01', '2024-01-01', freq='h', inclusive='left')
    pv = xr.DataArray(np.ones(len(time)), dims='time', coords={'time': time})
    site = SolarSite(capacity_kwp=1)
    costs = Economics(capex_inr_per_kwp=1000, import_inr_per_kwh=.2, export_inr_per_kwh=.1,
                       annual_opex_fraction=0, discount_rate=0, years=2)
    result = annual_summary(energy_balance(pv, pv*2), site, costs)
    assert result['net_annual_benefit_inr'] == pytest.approx(8760*.2)
    assert result['npv_inr'] == pytest.approx(-1000 + 2*8760*.2)
    zero = annual_summary(energy_balance(pv*0, pv*2), site, costs)
    assert zero['simple_payback_years'] is None and zero['self_consumption_fraction'] is None


def test_pvlib_demo_and_sensitivity():
    pytest.importorskip('pvlib')
    weather = make_solar_demo()
    assert weather.identical(make_solar_demo())
    pv = model_pv(weather)
    assert pv.min() >= 0 and pv.max() <= 100/1.1 + 1e-6
    night = (pd.DatetimeIndex(weather.time.values).hour == 0)
    assert (pv.values[night] == 0).all()
    cases = sensitivity_cases(pv, weather.demand_kw, SolarSite(), Economics())
    assert cases[1]['npv_inr'] > cases[0]['npv_inr'] > cases[2]['npv_inr']
    assert cases[3]['npv_inr'] < cases[0]['npv_inr'] < cases[4]['npv_inr']
    assert cases[5]['pv_kwh'] < cases[0]['pv_kwh'] < cases[6]['pv_kwh']


def test_solar_cli_exports_and_prepared_input(tmp_path):
    pytest.importorskip('pvlib')
    script = Path(__file__).resolve().parents[1]/'scripts/analyse_solar_potential.py'
    env = {**os.environ, 'MPLCONFIGDIR': str(tmp_path/'mpl')}
    subprocess.run([sys.executable, str(script), '--output-dir', str(tmp_path)],
                   env=env, check=True, capture_output=True, text=True)
    result = json.loads((tmp_path/'products/solar_energy_demo_summary.json').read_text())
    assert result['hours'] == 8760 and result['seed'] == 31
    assert (tmp_path/'figures/solar_energy_demo_balance.png').stat().st_size > 10000
    assert (tmp_path/'figures/solar_energy_demo_sensitivity.png').stat().st_size > 10000
    # Round-trip prepared hourly weather and load, not a claim of real observations.
    weather = make_solar_demo().to_dataframe()
    weather.index.name = 'timestamp_utc'
    input_path = tmp_path/'prepared.csv'
    weather.to_csv(input_path)
    subprocess.run([sys.executable, str(script), '--output-dir', str(tmp_path), '--input-csv', str(input_path)],
                   env=env, check=True, capture_output=True, text=True)
    prepared = json.loads((tmp_path/'products/solar_energy_input_summary.json').read_text())
    assert prepared['seed'] is None
    assert prepared['base_result']['pv_kwh'] == pytest.approx(result['base_result']['pv_kwh'])
