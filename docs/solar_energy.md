# From Kerala weather to solar-energy estimates

This optional follow-up asks how variable sunlight affects on-site solar use
and a simplified project cashflow. It stays in the existing Kerala setting:
an illustrative commercial building near Kochi (9.9312 N, 76.2673 E), not a
measured building or a surveyed roof. It is independent of the rainfall model;
rainfall predictions are not converted into solar radiation.

## Run it

From the repository root in the existing virtual environment:

```bash
python3 -m pip install -e '.[energy]'
python3 scripts/analyse_solar_potential.py
```

The optional dependency is pvlib; the original rainfall workflow is unchanged.
No online data download or API key is needed for the demo. To change the panel
capacity or keep an output run separately:

```bash
python3 scripts/analyse_solar_potential.py --capacity-kwp 50 --output-dir /tmp/kerala-solar
```

## How it works

1. Generate one complete UTC hourly year (2023, fixed seed 31). pvlib clear-sky
   irradiance is attenuated by synthetic cloud factors. Attenuation is deliberately
   stronger during June-September; that assumption is not an empirical finding.
   Temperature, wind and weekday daytime building demand are also simulated.
2. Use pvlib solar geometry and isotropic transposition to a south-facing,
   10-degree plane, followed by the SAPM cell-temperature model, PVWatts DC power
   and the PVWatts inverter model. The nominal array is 100 kWp, with 14% assumed
   non-inverter losses, a -0.004/degree C power coefficient and a 1.1 DC/AC ratio.
   Inverter clipping is applied before the AC energy balance.
3. For each hour, self-consumption = min(PV generation, demand). Surplus is
   exported, and the remaining demand is imported. No battery is modeled.
4. Calculate avoided import purchases plus export revenue, subtract assumed
   annual operating costs, and calculate simple payback and discounted NPV.
5. Change capital cost, import price and AC output one at a time. Recompute
   self-consumption for the output cases; do not scale financial benefits blindly.

Implementation references:
[pvlib PVWatts DC](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html),
[SAPM cell temperature](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.sapm_cell.html),
[PVWatts inverter](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html).

## Assumptions, not quotations

| Input | Illustrative value |
| --- | ---: |
| Installation cost | INR 60,000/kWp |
| Avoided import cost | INR 8.00/kWh |
| Export value | INR 3.00/kWh |
| Annual operating cost | 1.5% of initial capital |
| Evaluation horizon | 25 years |
| Real discount rate | 4% |

These are hypothetical scenario values, not current Kerala tariffs, supplier
quotes, net-metering entitlements or financial advice. Economics are defined in
the `Economics` dataclass; site assumptions are in `SolarSite`.

Annual net benefit = self-consumed kWh * import price + exported kWh * export
price - annual operating cost. NPV = -initial cost + the sum of discounted annual
net benefits, paid at each year end. Prices and generation are held constant in
real terms. Simple payback is initial cost / net annual benefit; nonpositive
benefit returns no payback. Payback beyond the horizon is not a claim of recovery
within the modeled life. No degradation, financing, replacement, tax, subsidy,
demand-charge benefit or residual value is included.

## Figures and products

- `figures/solar_energy_demo_balance.png`: a single June day and monthly balances.
- `figures/solar_energy_demo_sensitivity.png`: NPV sensitivity to explicit assumptions.
- `products/solar_energy_demo_report.md`: short interpretation and limitations.
- `products/solar_energy_demo_hourly.csv`: auditable hourly energy accounting.
- `products/solar_energy_demo_sensitivity.csv`: all case assumptions and results.
- `products/solar_energy_demo_summary.json`: parameters, versions and base result.

Hourly inputs are interval-average values on UTC interval-start timestamps;
solar geometry is calculated at the midpoint. Displays use Indian Standard Time
(`Asia/Kolkata`). Annual totals cover a full UTC year; monthly displays group
those same intervals by local month, including boundary intervals. The selected
15 June profile is an example day, not a representative-day statistic.

Self-consumption fraction is on-site PV use / PV generation; self-sufficiency is
on-site PV use / demand. They are not interchangeable. Each hourly balance
satisfies PV = on-site use + exports and demand = on-site use + imports.
Multiplying AC output by 0.85 or 1.15 is a stress test, not a calibrated weather
uncertainty interval or a new physical inverter simulation.

## Prepared weather and demand

```bash
python3 scripts/analyse_solar_potential.py --input-csv data/kochi_hourly_weather_load.csv
```

The CSV needs `timestamp_utc`, `ghi_wm2`, `dni_wm2`, `dhi_wm2`, `temp_air_c`,
`wind_speed_ms` and `demand_kw`. Use ISO UTC timestamps with a Z or +00:00 suffix,
one continuous complete UTC calendar year, finite values, irradiance in W/m2,
air temperature in degrees C, wind in m/s and average demand in kW. Missing
hours, duplicates and negative irradiance/demand are rejected, not silently
filled. Weather must describe the Kochi site unless `SolarSite` is explicitly
changed in code. Inputs require independently documented provenance, radiation
component consistency, interval conventions and demand quality control. Prepared
input is not automatically verified as measured data. Its output prefix is
`solar_energy_input`, so demo outputs are retained.

## Boundaries

No real weather or building data are fetched here. The simplified radiation and
load profiles cannot establish real energy yield, profitability or the financial
impact of monsoon rainfall. No measured-year climate variability, roof geometry,
local horizon/shading, snow/soiling model, export constraints, battery dispatch,
heating, emissions factors, CO2 savings or regulatory interpretation is supplied.
This is a transparent weather-to-energy workflow demonstration, not a bankable
yield assessment or a deployment-ready energy service.
