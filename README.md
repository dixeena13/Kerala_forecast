# Kerala Rainfall Forecast Postprocessing Prototype

Small application portfolio for the role **Wissenschaftliche/-r Mitarbeiter/-in im Rahmen der internationalen Entwicklungszusammenarbeit** at MeteoSchweiz.

This repository demonstrates how I would structure a practical ML workflow for improving local rainfall forecasts in **Kerala, India**, a South Asian monsoon region. It is a compact prototype, not an operational weather service: the included data are synthetic but shaped like meteorological station/NWP data so the full pipeline can be reviewed and run quickly.

## Why This Fits The MeteoSchweiz Role

The advertised role asks for meteorological data processing, ML for precipitation forecasts, operational software, deployment, international cooperation, and knowledge dissemination. This prototype mirrors that workflow:

- **South Asia focus:** Kerala is used as a concrete monsoon-rainfall use case.
- **Meteorological data pipeline:** `xarray` dataset with station metadata, latitude/longitude, elevation, coastal exposure, day of year, temperature, humidity, pressure, wind, NWP precipitation, and observed precipitation.
- **ML for rainfall:** PyTorch model for local precipitation postprocessing.
- **Operational mindset:** command-line training, validation metrics, FastAPI inference service, Dockerfile, and GitHub Actions CI.
- **Practical partner usability:** simple station endpoint, clear request schema, documented assumptions, and extension points for real netCDF/GRIB data.

## What The Prototype Does

1. Builds a Kerala monsoon-like demo dataset with `xarray`.
2. Converts meteorological variables into ML features.
3. Trains a small PyTorch model to postprocess raw NWP rainfall into local rainfall estimates.
4. Evaluates overall error and heavy-rain detection metrics.
5. Serves a `/predict` API for Kerala stations such as Kochi, Idukki, Wayanad, and Kozhikode.

## Repository Structure

```text
.
├── src/meteo_ml/
│   ├── kerala.py       # Kerala stations and metadata
│   ├── data.py         # xarray demo dataset and netCDF-ready loader
│   ├── features.py     # Feature extraction and standardization
│   ├── model.py        # PyTorch precipitation postprocessing model
│   ├── evaluate.py     # MAE/RMSE and heavy-rain metrics
│   ├── train.py        # Training and model artifact export
│   └── serve.py        # FastAPI inference service
├── tests/
│   └── test_pipeline.py
├── docs/
│   └── workflow.md     # Step-by-step explanation
├── .github/workflows/ci.yml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 -m pytest -o cache_dir=/tmp/meteo-ml-pytest-cache
python3 -m meteo_ml.train --epochs 20
python3 scripts/plot_kerala_demo.py
python3 scripts/plot_real_open_meteo_kochi.py
uvicorn meteo_ml.serve:app --reload
```

Example inference request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "station": "Kochi",
    "day_of_year": 180,
    "temperature": 298.0,
    "humidity": 0.86,
    "pressure": 1004.0,
    "wind_u": -5.0,
    "wind_v": 1.5,
    "nwp_precipitation": 28.0
  }'
```

Example response:

```json
{
  "station": "Kochi",
  "district": "Ernakulam",
  "precipitation_mm": 24.7,
  "heavy_rain_risk": "heavy",
  "model_version": "kerala-prototype-0.2.0"
}
```


## Figures

The repository includes two plotting scripts:

```bash
python3 scripts/plot_kerala_demo.py
python3 scripts/plot_real_open_meteo_kochi.py
```

The first figure visualizes the synthetic Kerala monsoon-like dataset used for the reproducible ML pipeline:

![Kerala demo rainfall](figures/kerala_demo_rainfall.png)

The second figure uses **real forecast data** from the Open-Meteo Forecast API for Kochi, Kerala. It shows daily precipitation sum and precipitation probability:

![Kochi real Open-Meteo forecast](figures/kochi_real_open_meteo_forecast.png)

The synthetic plot is useful for reproducible testing; the real-data plot shows how the repository can connect to live meteorological data without requiring a large GRIB/netCDF download.

## How I Would Extend This With Real Data

1. Replace the synthetic generator with real NWP forecasts and observations using `xarray`.
2. Add `cfgrib` ingestion for GRIB files and netCDF support for station/radar/reanalysis data.
3. Use spatial and temporal cross-validation, with special attention to heavy rainfall events.
4. Add model/version tracking, drift monitoring, and scheduled batch inference.
5. Package a lightweight deployment pattern that can run in cloud environments or at partner weather services with limited infrastructure.

## Application Note

This project is designed to show transferable strengths from physics, statistical modelling, uncertainty-aware time-series analysis, and Python-based scientific workflows. The Kerala framing makes the South Asia relevance explicit while keeping the scope achievable for a focused application portfolio.
