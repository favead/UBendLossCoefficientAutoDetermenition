from collections import OrderedDict
from typing import Union

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


def GP_regression_committee(
    regressors_params: list[dict[str, Union[int, float]]]
) -> list[GaussianProcessRegressor]:
    gp_regressors = []
    for regressor_params in regressors_params:
        gp_regressors.append(GP_regression(**regressor_params))
    return gp_regressors


def GS_x(labeled: np.ndarray, pool: np.ndarray) -> np.ndarray:
    distances = -2 * pool @ labeled.T
    distances = np.sum((pool**2.0), axis=1, keepdims=True) + distances
    distances = distances + np.sum((labeled**2.0), axis=1, keepdims=True).T
    distances_nx = distances[
        np.arange(distances.shape[0]), np.argmin(distances, axis=1)
    ]
    index = np.argmax(distances_nx)
    return index


def GS_y(labels: np.ndarray, pool_preds: np.ndarray) -> np.ndarray:
    distances = -2 * pool_preds @ labels.T
    distances = np.sum((pool_preds**2.0), axis=1, keepdims=True) + distances
    distances = distances + np.sum((labels**2.0), axis=1, keepdims=True).T
    distances_nx = distances[
        np.arange(distances.shape[0]), np.argmin(distances, axis=1)
    ]
    index = np.argmax(distances_nx)
    return index


def GS_xy(
    labeled: np.ndarray,
    pool: np.ndarray,
    labels: np.ndarray,
    pool_preds: np.ndarray,
) -> np.ndarray:
    d_X = -2 * pool @ labeled.T
    d_X = np.sum((pool**2.0), axis=1, keepdims=True) + d_X
    d_X = d_X + np.sum((labeled**2.0), axis=1, keepdims=True).T

    d_Y = -2 * pool_preds @ labels.T
    d_Y = np.sum((pool_preds**2.0), axis=1, keepdims=True) + d_Y
    d_Y = d_Y + np.sum((labels**2.0), axis=1, keepdims=True).T

    d_XY = d_X * d_Y
    d_nXY = d_XY[np.arange(d_XY.shape[0]), np.argmin(d_XY, axis=1)]
    index = np.argmax(d_nXY)
    return index
