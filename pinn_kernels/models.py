"""PINN models for the 1D heat equation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class HeatPINN(nn.Module):
    """MLP u(x, t). Tanh activations: the physics loss needs non-zero u_xx.

    If `learn_alpha` is set, the thermal diffusivity is a trainable parameter
    (inverse problem). It is stored as log(alpha) so it stays positive.
    """

    def __init__(self, width: int = 64, depth: int = 2, learn_alpha: bool = False,
                 alpha_init: float = 0.01):
        super().__init__()
        layers, d_in = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)
        self.log_alpha = nn.Parameter(torch.tensor([math.log(alpha_init)])) if learn_alpha else None

    @property
    def alpha(self) -> torch.Tensor | None:
        return None if self.log_alpha is None else self.log_alpha.exp()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))
