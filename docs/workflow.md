# Kerala Rainfall Prototype: Step-by-Step

Why Kerala: it is a South Asian monsoon region with coastal and orographic rainfall patterns

## Step 1: Create a meteorological demo dataset

The file `src/meteo_ml/data.py` creates a simulated `xarray.Dataset` with variables similar to a real workflow:

- station and district
- latitude and longitude
- elevation and distance to coast
- day of year
- temperature, humidity, pressure, wind components
- raw NWP precipitation
- observed precipitation target

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

## Step 5: Test and package

`tests/test_pipeline.py` checks the dataset, features, model shape, metrics, and API response. 

## Step 6: Production extensions

The next realistic extensions would be:

- real ERA5/IFS/local NWP ingestion with `xarray` and `cfgrib`
- station/radar validation data
- model artifact registry and versioning
- monitoring of input drift and forecast skill
- partner-facing documentation and lightweight deployment scripts
