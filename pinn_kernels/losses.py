"""Loss terms for the heat-equation PINN (forward and inverse problems)."""

from __future__ import annotations

import torch

from .residual import SPECS, fd_residual, interior


def ic_loss(model, n: int, L: float, gen: torch.Generator, device) -> torch.Tensor:
    x = torch.rand(n, 1, generator=gen, device=device) * L
    t = torch.zeros_like(x)
    u_true = torch.exp(-((x - L / 2) ** 2) / 0.1)
    return torch.mean((model(x, t) - u_true) ** 2)


def bc_loss(model, n: int, L: float, T: float, gen: torch.Generator, device) -> torch.Tensor:
    t = torch.rand(n, 1, generator=gen, device=device) * T
    left = model(torch.zeros_like(t), t)
    right = model(torch.full_like(t, L), t)
    return torch.mean(left ** 2) + torch.mean(right ** 2)


def data_loss(model, x, t, u) -> torch.Tensor:
    return torch.mean((model(x, t) - u) ** 2)


def physics_autograd(model, n: int, L: float, T: float, alpha, gen: torch.Generator, device) -> torch.Tensor:
    """Original approach: random collocation points, derivatives by autograd
    (second-order backprop through the MLP)."""
    x = (torch.rand(n, 1, generator=gen, device=device) * L).requires_grad_(True)
    t = (torch.rand(n, 1, generator=gen, device=device) * T).requires_grad_(True)
    u = model(x, t)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return torch.mean((u_t - alpha * u_xx) ** 2)


class GridSampler:
    """Uniform (nt, nx) space-time grid, optionally shifted by a random sub-cell
    jitter each step so the network is not only constrained at fixed nodes."""

    def __init__(self, nt: int, nx: int, L: float, T: float, device, jitter: bool = True):
        self.nt, self.nx, self.jitter, self.device = nt, nx, jitter, device
        self.dt, self.dx = T / (nt - 1), L / (nx - 1)
        t = torch.linspace(0.0, T, nt, device=device)
        x = torch.linspace(0.0, L, nx, device=device)
        self.T, self.X = torch.meshgrid(t, x, indexing="ij")   # (nt, nx), row-major in t

    def sample(self, gen: torch.Generator):
        T, X = self.T, self.X
        if self.jitter:
            s = torch.rand(2, generator=gen, device=self.device) - 0.5
            T = T + s[0] * self.dt
            X = X + s[1] * self.dx
        return X.reshape(-1, 1), T.reshape(-1, 1)


def physics_fd(model, grid: GridSampler, alpha, backend, gen: torch.Generator) -> torch.Tensor:
    """Kernel path: evaluate the network on the grid, compute the discrete
    residual with a kernel backend, backprop through its adjoint kernel."""
    x, t = grid.sample(gen)
    u = model(x, t).reshape(grid.nt, grid.nx)
    r = fd_residual(u, alpha, grid.dt, grid.dx, backend)
    return torch.mean(interior(r, SPECS[backend.pde]) ** 2)
