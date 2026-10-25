"""Export circulation-conditioned rainfall summaries and a diagnostic figure."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from meteo_ml.circulation import analyse_circulation, make_circulation_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Prepared daily netCDF, otherwise simulated demo.')
    parser.add_argument('--baseline-start', default='1991-01-01')
    parser.add_argument('--baseline-end', default='2005-12-31')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.input:
        with xr.open_dataset(args.input) as source:
            ds = source.load()
        provenance = f'User input: {args.input.name}; verify source metadata before interpretation.'
        stem = 'rainfall_circulation_input'
    else:
        ds = make_circulation_demo()
        provenance = ds.attrs['source']
        stem = 'rainfall_circulation_demo'
    records, annual = analyse_circulation(ds, args.baseline_start, args.baseline_end)
    (root / 'products').mkdir(exist_ok=True)
    (root / 'figures').mkdir(exist_ok=True)
    for suffix, rows in [('conditional', records), ('annual', annual)]:
        with (root / 'products' / f'{stem}_{suffix}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    colors = ['#007c91', '#bd4266', '#508230', '#8064a2']
    stations = list(ds.station.values)
    for i, station in enumerate(stations):
        color = colors[i % len(colors)]
        rows = [r for r in records if r['station'] == str(station)]
        width = .8 / len(stations)
        axes[0].bar(np.arange(3) - .4 + width*(i+.5),
                    [r['frequency_percent'] for r in rows], width, label=str(station), color=color)
        yearly = [r for r in annual if r['station'] == str(station)]
        axes[1].plot([r['year'] for r in yearly], [r['frequency_percent'] for r in yearly],
                     color=color, label=str(station), linewidth=1.4)
        axes[2].plot(np.arange(3), [r['valid_days'] for r in rows], 'o-', color=color)
    axes[0].set(title='Extreme rainfall by index bin', ylabel='Days above reference wet-day p95 (%)')
    axes[1].set(title='Year-to-year variability', xlabel='Year', ylabel='Paired valid days above threshold (%)')
    axes[2].set(title='Sample size matters', ylabel='Paired valid days')
    for ax in [axes[0], axes[2]]:
        ax.set_xticks(range(3), ['Lower third', 'Middle third', 'Upper third'])
        ax.set_xlabel('Reference circulation-index bins')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=9)
    fig.suptitle('Rainfall extremes and circulation | ' + ('prepared input' if args.input else 'simulated example'), fontsize=16)
    fig.supxlabel(f'{provenance}\nReference: {args.baseline_start} to {args.baseline_end}. Association is not causation.', fontsize=9)
    path = root / 'figures' / f'{stem}.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    report = f'''# Rainfall and circulation diagnostic

{provenance}

Reference period: {args.baseline_start} to {args.baseline_end}.
Extreme days exceed each station's reference 95th percentile on wet days (>=1 mm/day).
Index thirds are fitted to reference days only. Frequencies use paired valid days;
the CSVs retain denominators and event counts. Annual values describe variability,
not a fitted climate trend. No significance test or causal attribution is supplied.

Seasonality, persistence, missing data and short samples can affect associations.
The demo deliberately builds in a rainfall-index relationship. It does not provide
evidence about real Kerala circulation, Swiss weather, or future climate change.
'''
    (root / 'products' / f'{stem}_report.md').write_text(report)
    print(f'Created {path} and CSV/report files in {root / "products"}')


if __name__ == '__main__':
    main()
