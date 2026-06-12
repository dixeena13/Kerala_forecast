from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from meteo_ml.data import load_dataset
from meteo_ml.evaluate import regression_metrics
from meteo_ml.features import standardize, to_feature_matrix
from meteo_ml.model import PrecipitationRegressor


def train(epochs: int = 20, data_path: str | None = None, model_path: str | Path | None = None) -> dict[str, float]:
    ds = load_dataset(data_path)
    x_raw, y_raw = to_feature_matrix(ds)
    x_scaled, mean, std = standardize(x_raw)

    split = int(0.8 * len(x_scaled))
    x_train = torch.tensor(x_scaled[:split])
    y_train = torch.tensor(y_raw[:split])
    x_valid_np = x_scaled[split:]
    y_valid_np = y_raw[split:]

    model = PrecipitationRegressor(n_features=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.L1Loss()

    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(x_train)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_predictions = model(torch.tensor(x_valid_np)).detach().cpu().numpy()
    metrics = regression_metrics(y_valid_np, valid_predictions)

    if model_path is not None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_mean": mean,
                "feature_std": std,
                "metrics": metrics,
            },
            path,
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--model-path", default="artifacts/kerala_rain_model.pt")
    args = parser.parse_args()
    metrics = train(epochs=args.epochs, data_path=args.data_path, model_path=args.model_path)
    print(
        " ".join(
            [
                f"mae_mm={metrics['mae_mm']:.3f}",
                f"rmse_mm={metrics['rmse_mm']:.3f}",
                f"heavy_rain_recall={metrics['heavy_rain_recall']:.3f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
