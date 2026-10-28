"""Hand-written Triton kernels for the heat-equation residual and its adjoint.

Forward (from the original notebook, fixed and parameterized):
    r[t, x] = (u[t+1, x] - u[t, x]) / dt - alpha * (u[t, x+1] - 2 u[t, x] + u[t, x-1]) / dx^2
for t in [0, nt-1), x in [1, nx-1).

Adjoint (gather form over g zero-padded by 1):
    gu[t, x] = (-1/dt + 2a/dx^2) g[t, x] + (1/dt) g[t-1, x] - (a/dx^2) (g[t, x-1] + g[t, x+1])

Scalars are read from a device-side params tensor [dt, dx, alpha], so a
learnable alpha never forces a host sync.

Runs on CPU under TRITON_INTERPRET=1 (used by the test suite).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _heat_fwd_kernel(u_ptr, res_ptr, p_ptr, nt, nx, BLOCK_T: tl.constexpr, BLOCK_X: tl.constexpr):
    pid_t = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)
    dt = tl.load(p_ptr + 0)
    dx = tl.load(p_ptr + 1)
    alpha = tl.load(p_ptr + 2)

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = ((t < nt - 1)[:, None]) & (((x > 0) & (x < nx - 1))[None, :])

    base = t[:, None] * nx + x[None, :]
    u_c = tl.load(u_ptr + base, mask=mask, other=0.0)
    u_r = tl.load(u_ptr + base + 1, mask=mask, other=0.0)
    u_l = tl.load(u_ptr + base - 1, mask=mask, other=0.0)
    u_f = tl.load(u_ptr + base + nx, mask=mask, other=0.0)

    u_t = (u_f - u_c) / dt
    u_xx = ((u_r - 2.0 * u_c) + u_l) / (dx * dx)
    tl.store(res_ptr + base, u_t - alpha * u_xx, mask=mask)


@triton.jit
def _heat_adj_kernel(gpad_ptr, gu_ptr, p_ptr, nt, nx, BLOCK_T: tl.constexpr, BLOCK_X: tl.constexpr):
    pid_t = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)
    dt = tl.load(p_ptr + 0)
    dx = tl.load(p_ptr + 1)
    alpha = tl.load(p_ptr + 2)
    inv_dt = 1.0 / dt
    k = alpha / (dx * dx)

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = (t < nt)[:, None] & (x < nx)[None, :]

    W = nx + 2
    p = (t[:, None] + 1) * W + (x[None, :] + 1)   # g[t, x] in the padded buffer
    g_c = tl.load(gpad_ptr + p, mask=mask, other=0.0)
    g_up = tl.load(gpad_ptr + p - W, mask=mask, other=0.0)   # g[t-1, x]
    g_l = tl.load(gpad_ptr + p - 1, mask=mask, other=0.0)    # g[t, x-1]
    g_r = tl.load(gpad_ptr + p + 1, mask=mask, other=0.0)    # g[t, x+1]

    gu = (2.0 * k - inv_dt) * g_c + inv_dt * g_up - k * (g_l + g_r)
    tl.store(gu_ptr + t[:, None] * nx + x[None, :], gu, mask=mask)


def heat_forward(u: torch.Tensor, params: torch.Tensor, block_t: int = 8, block_x: int = 64) -> torch.Tensor:
    nt, nx = u.shape
    res = torch.zeros_like(u)
    grid = (triton.cdiv(nt, block_t), triton.cdiv(nx, block_x))
    _heat_fwd_kernel[grid](u, res, params, nt, nx, BLOCK_T=block_t, BLOCK_X=block_x)
    return res


def heat_adjoint(gpad: torch.Tensor, shape, params: torch.Tensor, block_t: int = 8, block_x: int = 64) -> torch.Tensor:
    nt, nx = shape
    gu = torch.empty((nt, nx), dtype=gpad.dtype, device=gpad.device)
    grid = (triton.cdiv(nt, block_t), triton.cdiv(nx, block_x))
    _heat_adj_kernel[grid](gpad, gu, params, nt, nx, BLOCK_T=block_t, BLOCK_X=block_x)
    return gu
