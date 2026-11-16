# Checking rainfall event proposals

The rainfall model answers "how much rain?" This independent extension asks
"did a proposed threshold exceedance coincide with a reference exceedance?"
It does not use the trained model, and it does not evaluate public warning bulletins.

## Run the demonstration

```bash
python3 scripts/verify_rainfall_warnings.py
python3 -m pytest tests/test_verification.py
```

The seeded example creates 120 daily samples at three Kerala locations and a
separate 8-by-6 illustrative latitude/longitude grid. Both reference rainfall and
forecast-like rainfall are simulated; the forecast-like series is constructed
from the reference plus prescribed bias and noise. This is a software check,
not an independent forecast experiment. Station and grid examples are separate
samples, not a station-to-grid interpolation or a comparison of their relative skill.

The script writes two figures, pooled and local CSV tables, a Markdown report,
and a JSON manifest with dates, dimensions, threshold and library versions.
Use `--threshold 30` to change the local-panel threshold, or `--output-dir /tmp/rain-check`
to save a run separately. Repeating a run overwrites files with the same prefix.

## Matching and scores

An event is rainfall **greater than or equal to** the chosen threshold. Default
diagnostics use 10, 20 and 40 mm in 24 hours. These are illustrative thresholds,
not warning levels from any meteorological service.

| Proposal | Reference event | Classification |
| --- | --- | --- |
| Yes | Yes | Hit (H) |
| No | Yes | Miss (M) |
| Yes | No | False alarm (F) |
| No | No | Correct negative (C) |

| Score | Formula | Interpretation |
| --- | --- | --- |
| Probability of detection (POD) | H / (H + M) | Fraction of reference events detected; higher is better |
| False alarm ratio (FAR) | F / (H + F) | Fraction of proposals not confirmed; lower is better |
| Critical success index (CSI) | H / (H + M + F) | Overlap after excluding correct negatives; higher is better |
| Frequency bias | (H + F) / (H + M) | Event frequency ratio; 1 means equal counts, not perfect timing |

Definitions follow the [forecast verification reference](https://jwgfvr.github.io/forecastverification/).
FAR is a false alarm **ratio**, not the false positive rate F/(F+C).
Zero denominators return NaN, including all-dry CSI, rather than an invented zero.
All counts, valid pairs and missing pairs remain available beside the scores.

Arrays must share explicit, unique and identical coordinates and dimension order.
The code never silently intersects dates or broadcasts a station series over a
grid. A pair is excluded if either value is NaN. Negative rainfall, infinity,
unsupported units and accumulation metadata are rejected.
Grid summaries give every paired cell-day equal weight. They are not area-weighted
regional scores. Counts are pooled before scores are calculated; averaging CSI
across stations/cells generally gives a different quantity.

## Prepared netCDF input

```bash
python3 scripts/verify_rainfall_warnings.py --input data/matched_daily_rainfall.nc --threshold 20
```

Provide `forecast_rainfall` and `reference_rainfall`, each with dimensions either
`(time, station)` or `(time, latitude, longitude)`. Both variables need
`units="mm"` and numeric `accumulation_hours=24` attributes. Time must be a sorted,
continuous daily Gregorian datetime coordinate; represent missing days with NaN.
Station identifiers or latitude/longitude coordinates must be explicit. A netCDF
backend appropriate to the file (for example `h5netcdf`) must be installed.
Input runs use `warning_verification_input_*` filenames, leaving demo outputs intact.

The caller must prepare the pairs: verify that forecast issue time precedes the
valid window, choose a fixed lead time, match the exact accumulation windows,
convert units and missing-value sentinels, quality-control observations, and
document grid/regridding and spatial representativeness. Declaring 24-hour
metadata is not proof these conditions hold. No archived forecasts or reference
observations are fetched automatically. Live Open-Meteo forecast plots elsewhere
in the repository are not an observation-based verification dataset.

For already matched issued-warning flags, use `verify_events` with 0/1/NaN arrays
and an independently defined reference event. It shares the same counting code.
This low-level interface does not parse bulletins or warning validity polygons.
It is hazard-neutral but no wind, snow or heat workflow is validated here.

## Scope and next scientific checks

Exact sample matching penalizes spatial displacement as both a miss and a false
alarm. This version has no neighborhood tolerance, episode matching, lead-time
stratification, uncertainty intervals, significance testing or impact verification.
Autocorrelation means cell-days are not independent trials. Rare-event scores
need event counts and a suitably long independent evaluation period; thresholds
should not be selected to improve scores on the same verification sample.
