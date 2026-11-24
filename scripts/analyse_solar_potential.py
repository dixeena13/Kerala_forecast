"""Generate an illustrative Kerala-site PV energy balance and sensitivity report."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meteo_ml.solar_energy import (Economics, SolarSite, energy_balance, make_solar_demo,
                                   model_pv, sensitivity_cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-kwp', type=float, default=100)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input-csv', type=Path, help='Complete UTC hourly year; schema in docs/solar_energy.md.')
    args = parser.parse_args()
    site, costs = SolarSite(capacity_kwp=args.capacity_kwp), Economics()
    source = 'SIMULATED weather and building demand | Kochi, Kerala example'
    prefix = 'solar_energy_demo'
    if args.input_csv:
        frame = pd.read_csv(args.input_csv)
        time = pd.DatetimeIndex(pd.to_datetime(frame.pop('timestamp_utc'), utc=True))
        frame.index = time.tz_localize(None)
        frame.index.name = 'time'
        weather = frame.to_xarray()
        source, prefix = f'Prepared input: {args.input_csv.name} | source not independently validated', 'solar_energy_input'
    else:
        weather = make_solar_demo(site)
    pv = model_pv(weather, site)
    balance = energy_balance(pv, weather.demand_kw)
    cases = sensitivity_cases(pv, weather.demand_kw, site, costs)
    base = cases[0]
    root = args.output_dir
    for name in ['products', 'figures']:
        (root/name).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cases).to_csv(root/'products'/f'{prefix}_sensitivity.csv', index=False)
    hourly = balance.to_dataframe()
    hourly.index = hourly.index.tz_localize('UTC')
    hourly.index.name = 'timestamp_utc'
    hourly.to_csv(root/'products'/f'{prefix}_hourly.csv')
    local = hourly.tz_convert('Asia/Kolkata')
    monthly = local.groupby(local.index.month).sum()
    # This is one illustrative local day, not a claim of a representative day.
    mask = (local.index.month == 6) & (local.index.day == 15)
    day = local.loc[mask]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    hours = day.index.hour + day.index.minute/60
    axes[0].plot(hours, day.pv_kwh, color='#087e8b', label='PV generation')
    axes[0].plot(hours, day.demand_kwh, color='#9b4360', label='Building demand')
    axes[0].fill_between(hours, day.self_consumed_kwh, color='#728831', alpha=.3, label='Self-consumed PV')
    axes[0].set(title='15 June | local hourly profile', xlabel='Interval start hour (IST)', ylabel='Energy per hour (kWh)', xlim=(0, 24))
    axes[0].legend(fontsize=9)
    axes[1].bar(monthly.index, monthly.self_consumed_kwh/1000, color='#087e8b', label='Self-consumed PV')
    axes[1].bar(monthly.index, monthly.export_kwh/1000, bottom=monthly.self_consumed_kwh/1000,
                color='#d0ab43', label='Exported PV')
    axes[1].plot(monthly.index, monthly.demand_kwh/1000, 'o-', color='#9b4360', label='Building demand')
    axes[1].set(title='Monthly energy balance', xlabel='Local calendar month', ylabel='Energy (MWh)', xticks=range(1, 13))
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Solar generation and on-site use | {site.capacity_kwp:g} kWp', fontsize=16)
    fig.supxlabel(source + '\nNo battery; synthetic demand in demo; no roof or shading assessment.', fontsize=9)
    fig.savefig(root/'figures'/f'{prefix}_balance.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    values = np.array([case['npv_inr']/1000 for case in cases])
    ax.barh([case['scenario'] for case in cases], values,
            color=['#9b4360' if value < 0 else '#087e8b' for value in values])
    ax.axvline(0, color='#444444', linewidth=.8)
    ax.axvline(base['npv_inr']/1000, color='#a48120', linestyle='--', label='Base NPV')
    ax.invert_yaxis()
    ax.set(xlabel='Net present value (thousand INR)', title='One-at-a-time financial sensitivity | 25 years, 4% discount rate')
    ax.legend()
    ax.grid(axis='x', alpha=.2)
    fig.supxlabel('Illustrative cost and tariff assumptions, not current market quotes.\nOutput multipliers are stress tests, not probabilities or measured weather-year variability.', fontsize=9)
    fig.savefig(root/'figures'/f'{prefix}_sensitivity.png', dpi=160)
    plt.close(fig)
    import pvlib
    manifest = {'source': source, 'site': asdict(site), 'economics': asdict(costs),
                'hours': len(weather.time), 'seed': None if args.input_csv else 31,
                'pvlib_version': pvlib.__version__, 'numpy_version': np.__version__,
                'pandas_version': pd.__version__, 'time': 'UTC hourly interval starts; geometry at midpoint',
                'cashflows': 'Constant annual output and real prices; end-of-year net benefit.',
                'base_result': base}
    (root/'products'/f'{prefix}_summary.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    payback = f"{base['simple_payback_years']:.1f} years" if base['simple_payback_years'] else 'not reached with nonpositive annual benefit'
    report = f'''# Solar-energy screening example

{source}

| Quantity | Result |
| --- | ---: |
| PV capacity | {site.capacity_kwp:g} kWp |
| Annual AC generation | {base['pv_kwh']/1000:.1f} MWh |
| Building demand | {base['demand_kwh']/1000:.1f} MWh |
| Self-consumed solar electricity | {base['self_consumed_kwh']/1000:.1f} MWh |
| Grid imports | {base['import_kwh']/1000:.1f} MWh |
| Grid exports | {base['export_kwh']/1000:.1f} MWh |
| Initial cost | INR {base['capex_inr']:,.0f} |
| Annual benefit after assumed O&M | INR {base['net_annual_benefit_inr']:,.0f} |
| Simple payback | {payback} |
| 25-year NPV at 4% | INR {base['npv_inr']:,.0f} |

Assumptions: INR 60,000/kWp installation cost; INR 8.00/kWh avoided import cost;
INR 3.00/kWh export revenue; annual O&M 1.5% of initial cost. These are hypothetical
inputs, not verified Kerala tariffs or quotations. Annual cashflows are held constant.
Sensitivity cases change one assumption at a time; output changes recalculate
self-consumption and exports, rather than scaling the financial benefit directly.

No subsidy, tax, financing, degradation, replacement, demand-charge savings,
battery, heating, export constraint, CO2 claim or regulatory analysis is included.
This is not a site-specific engineering design or investment recommendation.
See docs/solar_energy.md for method references and input requirements.
'''
    (root/'products'/f'{prefix}_report.md').write_text(report)
    print(f'Created {prefix} figures, hourly balance, sensitivity table and report in {root}')


if __name__ == '__main__':
    main()
