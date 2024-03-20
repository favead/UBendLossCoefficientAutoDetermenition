from collections import OrderedDict

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
