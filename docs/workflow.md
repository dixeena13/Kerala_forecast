# Kerala Weather Analytics: Workflow Guide

The project has three independent workflows: rainfall postprocessing, rainfall
event verification, and solar-energy screening. They share a Kerala setting and
scientific Python tooling, but are not an integrated operational service.

This page explains the original rainfall workflow. For the other analyses, see
the [verification guide](warning_verification.md) and the
[solar-energy guide](solar_energy.md). Solar generation is calculated from
radiation and temperature, not inferred from the rainfall model.

Why Kerala: it is a South Asian monsoon region with coastal and orographic rainfall patterns.

## Step 1: Create a meteorological demo dataset

The file `src/meteo_ml/data.py` creates a simulated `xarray.Dataset` with variables similar to a real workflow:

- station and district
- latitude and longitude
- elevation and distance to coast
- day of year
- temperature, humidity, pressure, wind components
- raw NWP precipitation
- simulated reference precipitation target (not a measured observation)

In an operational project, this layer would be replaced by real netCDF/GRIB/station/radar inputs.

## Step 2: Build ML-ready features

`src/meteo_ml/features.py` selects the meteorological predictors and standardizes them. This keeps data preparation separate from the model.

## Step 3: Train a PyTorch model

`src/meteo_ml/model.py` defines a small neural network. `src/meteo_ml/train.py` trains it and reports:

- MAE in mm
- RMSE in mm
- heavy-rain recall
- heavy-rain precision

## Step 4: Serve predictions through an API

`src/meteo_ml/serve.py` exposes:

- `GET /health`
- `GET /stations`
- `POST /predict`

The API trains its own small demo model at startup; it does not load the saved
offline checkpoint. The API serves rainfall predictions only, not solar-energy
or financial results.

## Step 5: Test and package

`tests/test_pipeline.py` checks the dataset, features, model shape, metrics, and API response. 

## Step 6: Production extensions

The next realistic extensions would be:

- real ERA5/IFS/local NWP ingestion with `xarray` and `cfgrib`
- station/radar validation data
- model artifact registry and versioning
- monitoring of input drift and forecast skill
- partner-facing documentation and lightweight deployment scripts
