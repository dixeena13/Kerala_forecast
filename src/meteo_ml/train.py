from __future__ import annotations

import argparse

import torch
from torch import nn

from meteo_ml.data import load_dataset
from meteo_ml.features import standardize, to_feature_matrix
from meteo_ml.model import PrecipitationRegressor


def train(epochs: int = 5, data_path: str | None = None) -> float:
    ds = load_dataset(data_path)
    x_raw, y_raw = to_feature_matrix(ds)
    x_scaled, _, _ = standardize(x_raw)

    x = torch.tensor(x_scaled)
    y = torch.tensor(y_raw)
    model = PrecipitationRegressor(n_features=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.L1Loss()

    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(x)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()

    return float(loss.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()
    loss = train(epochs=args.epochs, data_path=args.data_path)
    print(f"final_mae_mm={loss:.3f}")


if __name__ == "__main__":
    main()

