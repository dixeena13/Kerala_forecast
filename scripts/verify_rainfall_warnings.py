"""Run a reproducible rainfall-proposal verification or verify prepared netCDF pairs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import xarray as xr

from meteo_ml.verification import (classify_events, make_verification_demo,
                                   rainfall_events, verify_rainfall)


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_daily(ds):
    forecast, reference = ds.forecast_rainfall, ds.reference_rainfall
    if forecast.dims not in [('time', 'station'), ('time', 'latitude', 'longitude')]:
        raise ValueError('Expected (time, station) or (time, latitude, longitude).')
    time = ds.time.values
    if len(time) < 2 or not np.issubdtype(time.dtype, np.datetime64):
        raise ValueError('At least two Gregorian daily datetime coordinates required.')
    if not np.all(np.diff(time) == np.timedelta64(1, 'D')):
        raise ValueError('Time must be continuous daily, sorted and unique.')
    if forecast.dims != reference.dims:
        raise ValueError('Forecast and reference dimensions differ.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Prepared matched rainfall netCDF; no downloads.')
    parser.add_argument('--threshold', type=float, default=20, help='Illustrative mm/24h threshold.')
    parser.add_argument('--output-dir', type=Path, help='Default: repository root; separate input/demo filenames.')
    args = parser.parse_args()
    root = args.output_dir or Path(__file__).resolve().parents[1]
    if args.input:
        with xr.open_dataset(args.input) as opened:
            datasets = [opened.load()]
        prefix, source = 'warning_verification_input', f'Prepared input: {args.input.name}'
    else:
        datasets = make_verification_demo()
        prefix, source = 'warning_verification_demo', 'SIMULATED paired rainfall | seed 23'
    # Validate all inputs before creating any output files.
    for ds in datasets:
        validate_daily(ds)
        rainfall_events(ds.forecast_rainfall, ds.reference_rainfall, args.threshold)
    for directory in ['products', 'figures']:
        (root / directory).mkdir(parents=True, exist_ok=True)
    summary, locations = [], []
    for ds in datasets:
        layout = 'station' if 'station' in ds.dims else 'grid'
        f, o = ds.forecast_rainfall, ds.reference_rainfall
        thresholds = sorted(set([10., 20., 40., args.threshold]))
        for threshold in thresholds:
            scores = verify_rainfall(f, o, threshold, list(f.dims))
            summary.append({'layout': layout, 'threshold_mm_24h': threshold,
                            **{name: var.item() for name, var in scores.data_vars.items()}})
        local = verify_rainfall(f, o, args.threshold, ['time'])
        # xarray's stacked index retains station labels or each cell's coordinates.
        spatial = [dim for dim in f.dims if dim != 'time']
        stacked = local.stack(location=spatial)
        for i in range(stacked.sizes['location']):
            point = stacked.isel(location=i)
            locations.append({'layout': layout, 'threshold_mm_24h': args.threshold,
                              **{dim: point[dim].item() for dim in spatial},
                              **{name: var.item() for name, var in point.data_vars.items()}})
        outcome = classify_events(*rainfall_events(f, o, args.threshold))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
        selected = [r for r in summary if r['layout'] == layout]
        for metric, label, color in [('pod', 'Detection (POD)', '#087e8b'),
                                     ('far', 'False alarm ratio', '#bf4564'),
                                     ('csi', 'Critical success index', '#647b24')]:
            axes[0].plot([r['threshold_mm_24h'] for r in selected],
                         [r[metric] for r in selected], 'o-', label=label, color=color)
        axes[0].set(xlabel='Rainfall threshold (mm / 24 h)', ylabel='Score', ylim=(-.03, 1.03),
                    title='Pooled threshold diagnostics')
        axes[0].legend(fontsize=8, loc='best')
        axes[0].grid(alpha=.2)
        if layout == 'station':
            codes = outcome.transpose('station', 'time')
            image = axes[1].imshow(codes, aspect='auto', interpolation='nearest',
                                   cmap=ListedColormap(['#dce3e3', '#087e8b', '#bd404e', '#dcae39']),
                                   norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4))
            ticks = np.linspace(0, len(ds.time)-1, 4, dtype=int)
            axes[1].set_xticks(ticks, [str(ds.time.values[i])[:10] for i in ticks], rotation=25)
            axes[1].set_yticks(range(len(ds.station)), ds.station.values)
            axes[1].set_title('Daily event outcomes')
            bar = fig.colorbar(image, ax=axes[1], orientation='horizontal', pad=.08, ticks=range(4))
            bar.ax.set_xticklabels(['Correct\nnegative', 'Hit', 'Miss', 'False\nalarm'], fontsize=8)
            axes[2].bar(ds.station.values, local.csi.values, color='#087e8b')
            axes[2].set(title='Station critical success index', ylabel='CSI', ylim=(0, 1))
        else:
            for ax, name, title, cmap in [(axes[1], 'csi', 'Cell critical success index', 'viridis'),
                                          (axes[2], 'far', 'Cell false alarm ratio', 'magma')]:
                image = ax.pcolormesh(ds.longitude, ds.latitude, local[name], shading='auto',
                                      cmap=cmap, vmin=0, vmax=1)
                ax.set(title=title, xlabel='Longitude (degrees E)', ylabel='Latitude (degrees N)')
                fig.colorbar(image, ax=ax, shrink=.8)
        fig.suptitle(f'Rainfall proposal verification | {layout} samples', fontsize=16)
        fig.supxlabel(f'{source}\nLocal panels: >= {args.threshold:g} mm / 24 h. Missing pairs excluded; blank outcomes are missing, blank scores undefined.', fontsize=9)
        fig.savefig(root / 'figures' / f'{prefix}_{layout}.png', dpi=160)
        plt.close(fig)
    write_rows(root / 'products' / f'{prefix}_summary.csv', summary)
    # A unified schema handles the station and grid output rows without losing coordinates.
    fields = list(dict.fromkeys(key for row in locations for key in row))
    write_rows(root / 'products' / f'{prefix}_locations.csv',
               [{key: row.get(key, '') for key in fields} for row in locations])
    manifest = {'source': source, 'seed': None if args.input else 23,
                'threshold_mm_24h': args.threshold, 'comparison': '>=; exact paired coordinates',
                'pooling': 'unweighted counts, not mean of location scores',
                'undefined_scores': 'NaN in CSV',
                'numpy_version': np.__version__, 'xarray_version': xr.__version__,
                'samples': [{'dimensions': dict(ds.sizes), 'start': str(ds.time.values[0]),
                             'end': str(ds.time.values[-1])} for ds in datasets]}
    (root / 'products' / f'{prefix}_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    lines = ['# Rainfall proposal verification', '', source, '',
             '| Samples | Threshold (mm/24h) | Hits | Misses | False alarms | Valid pairs | POD | FAR | CSI |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary:
        lines.append('| ' + ' | '.join(str(row[k]) if k not in ['pod', 'far', 'csi'] else f'{row[k]:.3f}'
                                      for k in ['layout', 'threshold_mm_24h', 'hits', 'misses',
                                                'false_alarms', 'valid_pairs', 'pod', 'far', 'csi']) + ' |')
    lines.extend(['', 'Counts are pooled before division. NaN means the denominator is zero.',
                  'These are exact daily sample comparisons, not warning-episode or polygon verification.',
                  'Illustrative thresholds are not official warning criteria. Simulated scores are not real-world skill.',
                  'See docs/warning_verification.md for formulas, input requirements and limitations.', ''])
    (root / 'products' / f'{prefix}_report.md').write_text('\n'.join(lines))
    print(f'Wrote {prefix} figures and products under {root}')


if __name__ == '__main__':
    main()
