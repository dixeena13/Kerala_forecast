# Climate Scenario Extension

This follow-up extension connects the Kerala rainfall prototype to a climate-scenario product workflow.

## Indicators

The demo module `src/meteo_ml/scenarios.py` calculates:

- heavy-rain days per year above 20 mm/day
- mean annual maximum daily rainfall
- 95th percentile daily rainfall
- future-minus-baseline ensemble changes
- 10th-90th percentile ensemble spread

## Relation To Climate Scenario Work

This mirrors important parts of climate-scenario product support:

- translating climate-model output into usable indicators
- focusing on extremes and governing risk-relevant processes
- maintaining reproducible Python/xarray workflows
- communicating uncertainty to practitioners and end users

The included data are simulated for reproducibility and are not operational climate projections.
