# Kerala Rainfall Prototype: Step-by-Step

## Step 1: Define the use case

Goal: postprocess a raw numerical-weather-prediction rainfall forecast into a more local estimate for Kerala stations.

Why Kerala: it is a South Asian monsoon region with coastal and orographic rainfall patterns, so it is a relevant demonstration case for international meteorological cooperation.

## Step 2: Create a meteorological demo dataset

The file `src/meteo_ml/data.py` creates a synthetic `xarray.Dataset` with variables similar to a real workflow:

- station and district
- latitude and longitude
- elevation and distance to coast
- day of year
- temperature, humidity, pressure, wind components
- raw NWP precipitation
- observed precipitation target

In an operational project, this layer would be replaced by real netCDF/GRIB/station/radar inputs.

## Step 3: Build ML-ready features

`src/meteo_ml/features.py` selects the meteorological predictors and standardizes them. This keeps data preparation separate from the model, which is important for reproducibility and later deployment.

## Step 4: Train a PyTorch model

`src/meteo_ml/model.py` defines a small neural network. `src/meteo_ml/train.py` trains it and reports:

- MAE in mm
- RMSE in mm
- heavy-rain recall
- heavy-rain precision

Heavy-rain metrics matter because flood-relevant precipitation events are more operationally important than average error alone.

## Step 5: Serve predictions through an API

`src/meteo_ml/serve.py` exposes:

- `GET /health`
- `GET /stations`
- `POST /predict`

This demonstrates the deployment/MLOps side of the role: not only developing a model, but making it usable by another system or partner service.

## Step 6: Test and package

`tests/test_pipeline.py` checks the dataset, features, model shape, metrics, and API response. The Dockerfile and GitHub Actions workflow show how the project can be packaged and checked automatically.

## Step 7: Production extensions

The next realistic extensions would be:

- real ERA5/IFS/local NWP ingestion with `xarray` and `cfgrib`
- station/radar validation data
- model artifact registry and versioning
- monitoring of input drift and forecast skill
- partner-facing documentation and lightweight deployment scripts
