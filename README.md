# ML Precipitation Postprocessing Prototype

Small application portfolio for the role **Wissenschaftliche/-r Mitarbeiter/-in im Rahmen der internationalen Entwicklungszusammenarbeit** at MeteoSchweiz.

This repository shows how I would structure a practical, reproducible ML workflow for local precipitation forecast improvement: data ingestion, feature preparation, model training, evaluation, and a lightweight inference API. It is intentionally compact so the core ideas can be reviewed quickly.

## Why this is relevant to MeteoSchweiz

The advertised role combines meteorological data processing, machine learning, software engineering, deployment, and cooperation with international weather-service partners. This prototype focuses on the same working pattern:

- **Meteorological data pipeline:** `xarray`-based ingestion for gridded forecast/observation data, with a clear extension point for `netCDF` and `GRIB/cfgrib`.
- **ML for precipitation:** baseline PyTorch model for local precipitation postprocessing with uncertainty-aware evaluation hooks.
- **Operational mindset:** command-line training/evaluation, FastAPI inference endpoint, Dockerfile, and GitHub Actions CI.
- **Reproducibility:** typed Python modules, tests, dependency pinning, and documented assumptions.
- **Partner usability:** simple interfaces and plain documentation so methods can be discussed and adapted with domain partners.

## Repository Structure

```text
.
├── src/meteo_ml/
│   ├── data.py          # Synthetic xarray dataset and netCDF-ready loading API
│   ├── features.py      # Feature extraction from gridded meteorological fields
│   ├── model.py         # Small PyTorch precipitation postprocessing model
│   ├── train.py         # Training entry point
│   └── serve.py         # FastAPI inference service
├── tests/
│   └── test_pipeline.py # Smoke tests for data, model, and API schema
├── .github/workflows/ci.yml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python -m meteo_ml.train --epochs 3
uvicorn meteo_ml.serve:app --reload
```

Example inference request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 294.0, "humidity": 0.78, "pressure": 1008.0, "wind_u": 2.1, "wind_v": -0.7}'
```

## What I Would Extend Next

For a real MeteoSchweiz/South Asia project, I would extend this prototype in four steps:

1. Replace the synthetic generator with ERA5/IFS/local NWP and station/radar precipitation inputs using `xarray`, `cfgrib`, and `dask`.
2. Add spatial cross-validation and event-focused metrics for heavy precipitation, not only aggregate error.
3. Introduce model/version tracking and monitoring for data drift and inference quality.
4. Package deployment examples for cloud and local partner-weather-service environments.

## Application Note

My background is in physics, statistical modelling, machine learning, time-series analysis, uncertainty quantification, and Python-based scientific workflows. I have worked in Switzerland, India, the UK, and Belgium and am especially interested in applied climate and environmental data problems where scientific methods must become usable operational tools.

