from fastapi.testclient import TestClient
import torch

from meteo_ml.data import make_kerala_demo_dataset
from meteo_ml.evaluate import regression_metrics
from meteo_ml.features import FEATURE_NAMES, to_feature_matrix
from meteo_ml.model import PrecipitationRegressor
from meteo_ml.serve import app
from meteo_ml.train import train


def test_kerala_dataset_contains_expected_features() -> None:
    ds = make_kerala_demo_dataset(n_days=3)
    x, y = to_feature_matrix(ds)
    assert x.shape[1] == len(FEATURE_NAMES)
    assert y.shape[1] == 1
    assert ds.attrs["region"] == "Kerala, India / South Asia"


def test_model_forward_shape() -> None:
    ds = make_kerala_demo_dataset(n_days=2)
    x, _ = to_feature_matrix(ds)
    model = PrecipitationRegressor(n_features=x.shape[1])
    prediction = model(torch.tensor(x))
    assert prediction.shape == (x.shape[0], 1)


def test_training_returns_operational_metrics() -> None:
    metrics = train(epochs=2)
    assert metrics["mae_mm"] >= 0.0
    assert "heavy_rain_recall" in metrics


def test_metric_keys() -> None:
    metrics = regression_metrics(torch.tensor([[0.0], [25.0]]).numpy(), torch.tensor([[1.0], [30.0]]).numpy())
    assert set(metrics) == {"mae_mm", "rmse_mm", "heavy_rain_recall", "heavy_rain_precision"}


def test_api_prediction_for_kerala_station() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "station": "Kochi",
            "day_of_year": 180,
            "temperature": 298.0,
            "humidity": 0.86,
            "pressure": 1004.0,
            "wind_u": -5.0,
            "wind_v": 1.5,
            "nwp_precipitation": 28.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["district"] == "Ernakulam"
    assert payload["heavy_rain_risk"] in {"light-or-dry", "rain", "heavy", "very-heavy"}
