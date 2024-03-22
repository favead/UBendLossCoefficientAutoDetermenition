from collections import OrderedDict

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    WhiteKernel as Whtk,
    Matern as Matk,
)
import torch
from torch import nn


class PtLossRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.layers = self._create_linear_layer(0, input_dim, hidden_dim)
        for i in range(num_layers - 2):
            self.layers += self._create_linear_layer(
                i + 1, hidden_dim, hidden_dim
            )

        self.layers += self._create_linear_layer(
            num_layers - 1, hidden_dim, output_dim, is_last=True
        )
        self.layers = nn.Sequential(OrderedDict(self.layers))

    def _create_linear_layer(
        self, num: int, input_dim: int, output_dim: int, is_last: bool = False
    ) -> list[tuple[str, nn.Module]]:
        layers = [(f"lin_{num}", nn.Linear(input_dim, output_dim))]
        if not is_last:
            layers.append((f"tanh_{num}", nn.Tanh()))
        return layers

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)


def GP_regression(
    n_features: int,
    c_min: float,
    c_max: float,
    nu: float,
    ck_const: float,
    ck_min: float,
    ck_max: float,
    whtk_constant: float,
    whtk_min: float,
    whtk_max: float,
    n_restarts_optimizer: int,
) -> GaussianProcessRegressor:
    c_mean = [1.0] * n_features
    c_bounds = [[c_min, c_max]] * n_features
    kernel = C(ck_const, (ck_min, ck_max)) * Matk(c_mean, c_bounds, nu) + Whtk(
        whtk_constant, (whtk_min, whtk_max)
    )
    gp = GaussianProcessRegressor(
        kernel,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=False,
    )
    return gp


def GP_regression_std(
    GP_regression: GaussianProcessRegressor, X: np.ndarray
) -> np.ndarray:
    return np.argmax(GP_regression.predict(X, return_std=True))
