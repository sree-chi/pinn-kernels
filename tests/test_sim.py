import numpy as np

from pinn_kernels.sim import Domain, sensor_data, solve_heat


def test_fdm_matches_analytic_sine_mode():
    # u(x,0) = sin(pi x / L) has the exact solution exp(-alpha (pi/L)^2 t) sin(pi x / L)
    L, alpha = 2.0, 0.05
    sol = solve_heat(alpha=alpha, nx=201, domain=Domain(L=L, T=1.0), ic=lambda x: np.sin(np.pi * x / L))
    exact = np.exp(-alpha * (np.pi / L) ** 2 * sol.t[-1]) * np.sin(np.pi * sol.x / L)
    assert np.max(np.abs(sol.u[-1] - exact)) < 1e-4


def test_fdm_boundaries_and_decay():
    sol = solve_heat(alpha=0.05)
    assert np.all(sol.u[:, 0] == 0) and np.all(sol.u[:, -1] == 0)
    assert sol.u[-1].max() < sol.u[0].max()


def test_sensor_data_shapes_and_noise():
    sol = solve_heat(alpha=0.05, nx=101)
    x, t, u = sensor_data(sol, noise=0.0)
    assert x.shape == t.shape == u.shape == (21 * 3, 1)
    # noiseless readings at t=0 are the initial condition
    assert np.allclose(u[:3, 0].numpy(), np.exp(-((np.array([0.5, 1.0, 1.5]) - 1.0) ** 2) / 0.1), atol=1e-6)
