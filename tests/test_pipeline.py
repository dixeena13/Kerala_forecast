from fastapi.testclient import TestClient
import torch

from meteo_ml.data import make_synthetic_dataset
from meteo_ml.features import to_feature_matrix
from meteo_ml.model import PrecipitationRegressor
from meteo_ml.serve import app
from meteo_ml.train import train


def test_feature_pipeline_shapes() -> None:
    ds = make_synthetic_dataset(n_samples=12)
    x, y = to_feature_matrix(ds)
    assert x.shape == (12, 5)
    assert y.shape == (12, 1)


def test_model_forward_shape() -> None:
    ds = make_synthetic_dataset(n_samples=4)
    x, _ = to_feature_matrix(ds)
    model = PrecipitationRegressor(n_features=x.shape[1])
    prediction = model(torch.tensor(x))
    assert prediction.shape == (4, 1)


def test_training_smoke() -> None:
    loss = train(epochs=1)
    assert loss >= 0.0


def test_api_prediction() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"temperature": 294.0, "humidity": 0.78, "pressure": 1008.0, "wind_u": 2.1, "wind_v": -0.7},
    )
    assert response.status_code == 200
    assert "precipitation_mm" in response.json()
