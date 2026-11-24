"""Optional hourly PV screening using pvlib; no tariff or regulatory advice."""
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class SolarSite:
    latitude: float = 9.9312
    longitude: float = 76.2673
    altitude_m: float = 10
    capacity_kwp: float = 100
    tilt_deg: float = 10
    azimuth_deg: float = 180
    losses_fraction: float = .14
    dc_ac_ratio: float = 1.1

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Site parameters must be finite.')
        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError('Invalid coordinates.')
        if not (self.capacity_kwp > 0 and self.dc_ac_ratio > 0 and 0 <= self.losses_fraction < 1):
            raise ValueError('Invalid capacity, inverter ratio or loss fraction.')
        if not (0 <= self.tilt_deg <= 90 and 0 <= self.azimuth_deg < 360):
            raise ValueError('Invalid panel orientation.')


@dataclass(frozen=True)
class Economics:
    capex_inr_per_kwp: float = 60000
    import_inr_per_kwh: float = 8.0
    export_inr_per_kwh: float = 3.0
    annual_opex_fraction: float = .015
    discount_rate: float = .04
    years: int = 25

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Economic assumptions must be finite.')
        if min(self.capex_inr_per_kwp, self.import_inr_per_kwh, self.export_inr_per_kwh,
               self.annual_opex_fraction, self.discount_rate) < 0:
            raise ValueError('Economic assumptions must be nonnegative.')
        if self.capex_inr_per_kwp == 0 or not isinstance(self.years, int) or self.years < 1:
            raise ValueError('Positive capex and an integer horizon are required.')


def _pvlib():
    try:
        import pvlib
    except ImportError as exc:
        raise ImportError("Install the optional solar tools: python3 -m pip install -e '.[energy]'") from exc
    return pvlib


def hourly_index(time) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(time)
    if index.tz is None:
        index = index.tz_localize('UTC')
    else:
        index = index.tz_convert('UTC')
    if len(index) < 2 or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError('Time must be unique, ordered UTC hours.')
    if not np.all((index[1:] - index[:-1]) == pd.Timedelta(hours=1)):
        raise ValueError('Inputs must have continuous one-hour intervals.')
    return index


def make_solar_demo(site: SolarSite = SolarSite(), year: int = 2023, seed: int = 31) -> xr.Dataset:
    """Clear-sky geometry plus synthetic cloud attenuation, temperature and demand."""
    pvlib = _pvlib()
    time = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', freq='h', inclusive='left', tz='UTC')
    midpoint = time + pd.Timedelta(minutes=30)
    location = pvlib.location.Location(site.latitude, site.longitude, altitude=site.altitude_m)
    clear = location.get_clearsky(midpoint)
    position = location.get_solarposition(midpoint)
    rng = np.random.default_rng(seed)
    cloud = np.repeat(rng.beta(3, 2, len(time)//24), 24)
    # Prescribed monsoon attenuation is illustrative, not fitted to observations.
    monsoon = (time.month >= 6) & (time.month <= 9)
    cloud *= np.where(monsoon, .6, 1.0)
    ghi = clear.ghi.to_numpy() * (.25 + .75*cloud)
    dni = clear.dni.to_numpy() * cloud**2
    cosine = np.maximum(0, np.cos(np.deg2rad(position.apparent_zenith)))
    dhi = np.maximum(0, ghi - dni*cosine)
    local = midpoint.tz_convert('Asia/Kolkata')
    temperature = 28 + 2*np.cos(2*np.pi*(time.dayofyear.to_numpy()-105)/365.25)
    temperature += 3*np.sin(2*np.pi*(local.hour.to_numpy()-8)/24) + rng.normal(0, 1.5, len(time))
    work = (local.dayofweek < 5) & (local.hour >= 8) & (local.hour < 18)
    demand = np.maximum(0, 12 + 30*work + rng.normal(0, 1.2, len(time)))
    fields = {'ghi_wm2': ghi, 'dni_wm2': dni, 'dhi_wm2': dhi,
              'temp_air_c': temperature, 'wind_speed_ms': rng.uniform(.5, 4, len(time)), 'demand_kw': demand}
    return xr.Dataset({name: ('time', np.asarray(values)) for name, values in fields.items()},
                      coords={'time': time.tz_localize(None).to_numpy()},
                      attrs={'source': 'Simulated weather and commercial demand; not measured site data.',
                             'timezone': 'UTC', 'seed': seed})


def model_pv(weather: xr.Dataset, site: SolarSite = SolarSite()) -> xr.DataArray:
    """Return average AC kW for hourly interval-start timestamps (UTC)."""
    pvlib = _pvlib()
    time = hourly_index(weather.time.values)
    names = ['ghi_wm2', 'dni_wm2', 'dhi_wm2', 'temp_air_c', 'wind_speed_ms']
    for name in names:
        if weather[name].dims != ('time',) or not np.isfinite(weather[name].values).all():
            raise ValueError(f'{name} must be finite with dimension time; no missing-hour filling.')
        if name != 'temp_air_c' and bool((weather[name] < 0).any()):
            raise ValueError(f'{name} cannot be negative.')
    midpoint = time + pd.Timedelta(minutes=30)
    position = pvlib.solarposition.get_solarposition(midpoint, site.latitude, site.longitude,
                                                   altitude=site.altitude_m)
    poa = pvlib.irradiance.get_total_irradiance(
        site.tilt_deg, site.azimuth_deg, position.apparent_zenith.to_numpy(),
        position.azimuth.to_numpy(), weather.dni_wm2.values, weather.ghi_wm2.values,
        weather.dhi_wm2.values, model='isotropic')['poa_global']
    cell = pvlib.temperature.sapm_cell(poa, weather.temp_air_c.values, weather.wind_speed_ms.values,
                                       a=-3.56, b=-.075, deltaT=3)
    dc = pvlib.pvsystem.pvwatts_dc(poa, cell, site.capacity_kwp*1000, gamma_pdc=-.004)
    dc = np.maximum(0, dc) * (1-site.losses_fraction)
    # Inverter pdc0 is its DC input rating, distinct from the module DC capacity.
    inverter_input = site.capacity_kwp*1000/site.dc_ac_ratio/.96
    ac = np.maximum(0, pvlib.inverter.pvwatts(dc, pdc0=inverter_input, eta_inv_nom=.96))/1000
    ac = np.where(position.apparent_zenith.to_numpy() >= 90, 0, ac)
    if not np.isfinite(ac).all():
        raise ValueError('Nonfinite PV result; inspect weather and geometry.')
    return xr.DataArray(ac, dims='time', coords={'time': weather.time}, name='pv_kw', attrs={'units': 'kW'})


def energy_balance(pv_kw: xr.DataArray, demand_kw: xr.DataArray) -> xr.Dataset:
    """Hourly kW * one hour = kWh. No storage, curtailment or export limit."""
    if pv_kw.dims != ('time',) or demand_kw.dims != ('time',):
        raise ValueError('PV and demand must have dimension time only.')
    xr.align(pv_kw, demand_kw, join='exact')
    hourly_index(pv_kw.time.values)
    for array in [pv_kw, demand_kw]:
        if not np.isfinite(array.values).all() or bool((array < 0).any()):
            raise ValueError('PV and demand must be finite and nonnegative.')
    consumed = np.minimum(pv_kw, demand_kw)
    result = xr.Dataset({'pv_kwh': pv_kw, 'demand_kwh': demand_kw,
                       'self_consumed_kwh': consumed, 'export_kwh': pv_kw-consumed,
                       'import_kwh': demand_kw-consumed})
    for array in result.data_vars.values():
        array.attrs = {'units': 'kWh', 'interval_hours': 1}
    return result


def annual_summary(balance: xr.Dataset, site: SolarSite, costs: Economics) -> dict:
    time = hourly_index(balance.time.values)
    year = time[0].year
    expected = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', freq='h', inclusive='left', tz='UTC')
    if not time.equals(expected):
        raise ValueError('Annual economics requires one complete UTC calendar year, not annualized partial data.')
    totals = {name: float(array.sum()) for name, array in balance.data_vars.items()}
    capex = site.capacity_kwp*costs.capex_inr_per_kwp
    gross = totals['self_consumed_kwh']*costs.import_inr_per_kwh + totals['export_kwh']*costs.export_inr_per_kwh
    net = gross - capex*costs.annual_opex_fraction
    npv = -capex + sum(net/(1+costs.discount_rate)**year for year in range(1, costs.years+1))
    return {**totals, 'self_consumption_fraction': totals['self_consumed_kwh']/totals['pv_kwh'] if totals['pv_kwh'] else None,
            'self_sufficiency_fraction': totals['self_consumed_kwh']/totals['demand_kwh'] if totals['demand_kwh'] else None,
            'capex_inr': capex, 'gross_annual_benefit_inr': gross,
            'net_annual_benefit_inr': net, 'simple_payback_years': capex/net if net > 0 else None,
            'npv_inr': npv}


def sensitivity_cases(pv_kw, demand_kw, site, costs) -> list[dict]:
    cases = [('Base assumptions', 1., costs),
             ('Capital cost -20%', 1., replace(costs, capex_inr_per_kwp=costs.capex_inr_per_kwp*.8)),
             ('Capital cost +20%', 1., replace(costs, capex_inr_per_kwp=costs.capex_inr_per_kwp*1.2)),
             ('Import price -25%', 1., replace(costs, import_inr_per_kwh=costs.import_inr_per_kwh*.75)),
             ('Import price +25%', 1., replace(costs, import_inr_per_kwh=costs.import_inr_per_kwh*1.25)),
             ('AC output -15%', .85, costs), ('AC output +15%', 1.15, costs)]
    return [{'scenario': label, 'output_multiplier': scale, **asdict(case_costs),
             **annual_summary(energy_balance(pv_kw*scale, demand_kw), site, case_costs)}
            for label, scale, case_costs in cases]
