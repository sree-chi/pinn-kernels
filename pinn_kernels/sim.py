"""Classical finite-difference solver for the 1D heat equation (ground truth).

u_t = alpha * u_xx on x in [0, L], t in [0, T]
u(x, 0) = exp(-(x - L/2)^2 / 0.1), u(0, t) = u(L, t) = 0

Explicit FTCS scheme. Stability requires alpha * dt / dx^2 <= 1/2; we use 1/4
like the original notebook. Vectorized over x (the notebook looped in Python).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Domain:
    L: float = 2.0
    T: float = 1.0


def gaussian_ic(x: np.ndarray, L: float = 2.0) -> np.ndarray:
    return np.exp(-((x - L / 2) ** 2) / 0.1)


@dataclass
class FDMSolution:
    x: np.ndarray          # (nx,)
    t: np.ndarray          # (n_steps + 1,)
    u: np.ndarray          # (n_steps + 1, nx)

    def at(self, t_query: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the stored space-time field."""
        t_query = np.clip(np.asarray(t_query, dtype=np.float64), self.t[0], self.t[-1])
        x_query = np.clip(np.asarray(x_query, dtype=np.float64), self.x[0], self.x[-1])
        it = np.clip(np.searchsorted(self.t, t_query) - 1, 0, len(self.t) - 2)
        ix = np.clip(np.searchsorted(self.x, x_query) - 1, 0, len(self.x) - 2)
        wt = (t_query - self.t[it]) / (self.t[it + 1] - self.t[it])
        wx = (x_query - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        u00 = self.u[it, ix]
        u01 = self.u[it, ix + 1]
        u10 = self.u[it + 1, ix]
        u11 = self.u[it + 1, ix + 1]
        return (1 - wt) * ((1 - wx) * u00 + wx * u01) + wt * ((1 - wx) * u10 + wx * u11)


def solve_heat(alpha: float = 0.05, nx: int = 201, domain: Domain = Domain(), ic=gaussian_ic,
               cfl: float = 0.25) -> FDMSolution:
    x = np.linspace(0.0, domain.L, nx)
    dx = x[1] - x[0]
    dt_target = cfl * dx * dx / alpha
    n_steps = int(np.ceil(domain.T / dt_target))
    dt = domain.T / n_steps
    r = alpha * dt / (dx * dx)
    assert r <= 0.5 + 1e-12, f"unstable FTCS step (r={r:.3f})"

    u = np.empty((n_steps + 1, nx))
    u[0] = ic(x, domain.L) if ic is gaussian_ic else ic(x)
    u[0, 0] = u[0, -1] = 0.0
    for n in range(n_steps):
        prev = u[n]
        nxt = u[n + 1]
        nxt[1:-1] = prev[1:-1] + r * (prev[2:] - 2.0 * prev[1:-1] + prev[:-2])
        nxt[0] = nxt[-1] = 0.0
    t = np.linspace(0.0, domain.T, n_steps + 1)
    return FDMSolution(x=x, t=t, u=u)


def sensor_data(sol: FDMSolution, locations=(0.5, 1.0, 1.5), every: float = 0.05,
                noise: float = 0.10, seed: int = 0):
    """Synthetic noisy sensor readings sampled from the FDM field.

    Noise is Gaussian with std = noise * std(clean readings), as in the notebook.
    """
    times = np.arange(0.0, sol.t[-1] + 1e-9, every)
    tt, xx = np.meshgrid(times, np.asarray(locations), indexing="ij")
    clean = sol.at(tt.ravel(), xx.ravel())
    rng = np.random.default_rng(seed)
    noisy = clean + noise * clean.std() * rng.standard_normal(clean.shape)
    as_t = lambda a: torch.tensor(a, dtype=torch.float32).view(-1, 1)
    return as_t(xx.ravel()), as_t(tt.ravel()), as_t(noisy)
