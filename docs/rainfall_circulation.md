# What accompanies the wettest days?

This follow-up keeps the Kerala rainfall focus and adds a small descriptive
analysis: are very wet days more frequent in different parts of a daily
circulation index distribution?

## Run it

From the repository root, after the usual editable installation:

```bash
python3 scripts/analyse_rainfall_circulation.py
python3 -m pytest tests/test_circulation.py
```

1. Generate seeded daily rainfall and a persistent toy index for 1991-2020.
2. Fit station wet-day rainfall percentiles and index bins on 1991-2005 only.
3. Count threshold exceedances by index bin and by calendar year.
4. Export CSVs, a short methods report and a three-panel diagnostic figure.

Wet days mean at least 1 mm/day. The threshold is the reference wet-day 95th
percentile; event counts here are not the ETCCDI R95pTOT rainfall-total index.
See the [Copernicus index definitions](https://surfobs.climate.copernicus.eu/userguidance/indicesdictionary.php).

## Prepared real-data input

```bash
python3 scripts/analyse_rainfall_circulation.py --input data/daily_analysis.nc --baseline-start 1991-01-01 --baseline-end 2005-12-31
```

The netCDF must contain `precipitation(time, station)` with units `mm/day`,
`circulation_index(time)`, named stations and a continuous Gregorian daily time
coordinate. Missing values may be NaN, but dates must be present. Convert units,
align accumulation windows and calendars, and document the original data source
before running. netCDF4/HDF5 inputs may need an additional xarray backend such as
`h5netcdf`; the demo itself needs no extra dependencies. No data are downloaded.

For real work, the index must be physically justified and independently derived
from circulation fields or a documented index source. This tool does not derive
pressure patterns, identify blocking, or estimate a teleconnection. Input outputs
use a separate filename prefix so they do not overwrite simulated results.

## Interpretation limits

The demo intentionally amplifies rainfall for positive index values. Recovering
that association checks the workflow, not a scientific hypothesis. Bins are
statistical index thirds, not named weather regimes. Conditional frequencies are
descriptive: seasonality and serial dependence require further treatment before
formal inference. Annual curves show variability, not evidence for a trend.
No pre-industrial reconstruction, future projection, attribution, or operational
warning is produced. Sample counts and missing-data handling are exposed so a
reviewer can inspect what went into each result.
