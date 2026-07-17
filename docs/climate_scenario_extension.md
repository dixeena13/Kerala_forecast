# Climate Scenario Extension

This follow-up extension connects the Kerala rainfall prototype to a climate-scenario product workflow.

The aim is not to reproduce official Swiss climate scenarios. Instead, it demonstrates the type of technical workflow that appears in climate-scenario product development:

1. Create or ingest ensemble climate data.
2. Derive user-facing extreme-rainfall indicators.
3. Summarize ensemble spread and uncertainty.
4. Produce a compact figure that can be interpreted by users outside the modelling team.

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
