"""Finite-difference PDE residual on a grid, with interchangeable kernel backends.

This is the bridge between the PINN and the kernels. The network is evaluated
on an (nt, nx) space-time grid; the residual of the discretized PDE is then
computed by a kernel, and gradients flow back to the network (and to alpha in
the inverse problem) through a custom autograd.Function whose backward pass
calls the adjoint kernel:

    u = model(grid)                    # (nt, nx), autograd tracks the MLP
    r = fd_residual(u, alpha, backend)  # kernel forward
    loss = mean(r_interior ** 2)
    loss.backward()                    # adjoint kernel -> dL/du -> MLP

Backends:
    torch   reference implementation (tensor slicing; autograd for the VJP)
    llvm    CPU kernels JIT-compiled from the stencil graph (csrc/codegen.cpp)
    triton  hand-written Triton GPU kernels (heat equation only)
    ptx     GPU kernels emitted by LLVM's NVPTX backend from the same graph,
            loaded with CuPy

Parameter vector layout for every backend: [dt, dx, <physical params>].
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class Spec:
    params: tuple[str, ...]
    physics: tuple[str, ...]
    t_lo: int = 0  # interior: t in [t_lo, nt - t_hi), x in [x_lo, nx - x_hi)
    t_hi: int = 1
    x_lo: int = 1
    x_hi: int = 1
    halo: int = 1


SPECS = {
    "heat": Spec(params=("dt", "dx", "alpha"), physics=("alpha",)),
    "burgers": Spec(params=("dt", "dx", "nu"), physics=("nu",)),
}


def interior_mask(u: torch.Tensor, spec: Spec) -> torch.Tensor:
    m = torch.zeros_like(u)
    nt, nx = u.shape
    m[spec.t_lo:nt - spec.t_hi, spec.x_lo:nx - spec.x_hi] = 1
    return m


def interior(r: torch.Tensor, spec: Spec) -> torch.Tensor:
    nt, nx = r.shape
    return r[spec.t_lo:nt - spec.t_hi, spec.x_lo:nx - spec.x_hi]


# ---------------------------------------------------------------------------
# Reference (pure PyTorch) residuals
# ---------------------------------------------------------------------------
def reference_residual(pde: str, u: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Residual on the full grid (zeros outside the interior). Differentiable."""
    dt, dx, c = params[0], params[1], params[2]
    uc, uf = u[:-1, 1:-1], u[1:, 1:-1]
    ur, ul = u[:-1, 2:], u[:-1, :-2]
    u_t = (uf - uc) / dt
    u_xx = ((ur - 2.0 * uc) + ul) / (dx * dx)
    if pde == "heat":
        r = u_t - c * u_xx
    elif pde == "burgers":
        r = (u_t + uc * ((ur - ul) / (2.0 * dx))) - c * u_xx
    else:
        raise ValueError(pde)
    return F.pad(r, (1, 1, 0, 1))


class TorchBackend:
    name = "torch"

    def __init__(self, pde: str = "heat"):
        self.pde, self.spec = pde, SPECS[pde]

    def forward(self, u, params):
        with torch.no_grad():
            return reference_residual(self.pde, u, params)

    def _vjp(self, u, g, params):
        with torch.enable_grad():
            u_ = u.detach().requires_grad_(True)
            p_ = params.detach().requires_grad_(True)
            r = reference_residual(self.pde, u_, p_)
            gu, gp = torch.autograd.grad(r, (u_, p_), grad_outputs=g)
        return gu, gp

    def adjoint(self, u, g, params):
        return self._vjp(u, g, params)[0]

    def param_grad(self, u, g, params):
        return self._vjp(u, g, params)[1][2:]


class LLVMBackend:
    """CPU kernels JIT-compiled by LLVM from the stencil expression graph."""

    name = "llvm"

    def __init__(self, pde: str = "heat", opt: int = 3, fast: bool = False):
        from ._jit import JitKernels

        self.pde, self.spec = pde, SPECS[pde]
        self.k = JitKernels(pde, opt=opt, fast=fast)
        s = self.spec
        if (self.k.t_lo, self.k.t_hi, self.k.x_lo, self.k.x_hi, self.k.halo) != (s.t_lo, s.t_hi, s.x_lo, s.x_hi, s.halo):
            raise RuntimeError("LLVM stencil bounds disagree with the Python spec")

    def forward(self, u, params):
        return self.k.forward(u.contiguous(), params.contiguous())

    def adjoint(self, u, g, params):
        h = self.k.halo
        gpad = F.pad(g, (h, h, h, h)).contiguous()
        return self.k.adjoint(gpad, u.contiguous(), params.contiguous())

    def param_grad(self, u, g, params):
        vals = [self.k.param_grad(i, u.contiguous(), g.contiguous(), params.contiguous())
                for i in range(len(self.k.physics_params))]
        return torch.tensor(vals, dtype=params.dtype)


class TritonBackend:
    """Hand-written Triton kernels (heat equation)."""

    name = "triton"

    def __init__(self, pde: str = "heat", block_t: int = 8, block_x: int = 64):
        if pde != "heat":
            raise ValueError("Triton kernels are hand-written for the heat equation only; use llvm/ptx for others")
        from . import triton_kernels

        self.pde, self.spec = pde, SPECS[pde]
        self.tk, self.block_t, self.block_x = triton_kernels, block_t, block_x
        self._ref = TorchBackend(pde)

    def forward(self, u, params):
        return self.tk.heat_forward(u.contiguous(), params.contiguous(), self.block_t, self.block_x)

    def adjoint(self, u, g, params):
        gpad = F.pad(g, (1, 1, 1, 1)).contiguous()
        return self.tk.heat_adjoint(gpad, u.shape, params.contiguous(), self.block_t, self.block_x)

    def param_grad(self, u, g, params):
        # dR/dalpha = -u_xx: a cheap fused reduction in PyTorch
        return self._ref.param_grad(u, g, params)


class PTXBackend:
    """GPU kernels emitted by LLVM's NVPTX backend from the stencil graph."""

    name = "ptx"

    def __init__(self, pde: str = "heat", fast: bool = False, block=(32, 8)):
        import cupy  # noqa: F401  (optional dependency, GPU only)

        from ._jit import emit_ptx

        self.cp = cupy
        self.pde, self.spec, self.block = pde, SPECS[pde], block
        major, minor = torch.cuda.get_device_capability()
        sm = min(major * 10 + minor, 90)  # LLVM 18 knows up to sm_90; the driver JITs PTX forward
        self.ptx = emit_ptx(pde, f"sm_{sm}", fast=fast)
        fd, path = tempfile.mkstemp(suffix=".ptx")
        with os.fdopen(fd, "w") as f:
            f.write(self.ptx)
        self.module = cupy.RawModule(path=path)
        self.fwd = self.module.get_function("pk_fwd")
        self.adj = self.module.get_function("pk_adj")
        self._ref = TorchBackend(pde)

    def _launch(self, fn, shape, args):
        import numpy as np

        nt, nx = shape
        bx, by = self.block
        grid = ((nx + bx - 1) // bx, (nt + by - 1) // by, 1)
        cp = self.cp
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            fn(grid, (bx, by, 1), tuple(cp.from_dlpack(a) if isinstance(a, torch.Tensor) else a for a in args)
               + (np.int64(nt), np.int64(nx)))

    def forward(self, u, params):
        u, params = u.contiguous(), params.contiguous()
        out = torch.zeros_like(u)
        self._launch(self.fwd, u.shape, (u, out, params))
        return out

    def adjoint(self, u, g, params):
        u, params = u.contiguous(), params.contiguous()
        gpad = F.pad(g, (1, 1, 1, 1)).contiguous()
        out = torch.empty_like(u)
        self._launch(self.adj, u.shape, (gpad, out, u, params))
        return out

    def param_grad(self, u, g, params):
        return self._ref.param_grad(u, g, params)


def get_backend(name: str, pde: str = "heat", **kw):
    return {"torch": TorchBackend, "llvm": LLVMBackend, "triton": TritonBackend, "ptx": PTXBackend}[name](pde, **kw)


# ---------------------------------------------------------------------------
# Autograd bridge
# ---------------------------------------------------------------------------
class _FDResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, phys, dt, dx, backend):
        params = torch.cat([torch.tensor([dt, dx], dtype=u.dtype, device=u.device),
                            phys.detach().to(u.dtype).reshape(-1)])
        res = backend.forward(u.detach(), params)
        ctx.save_for_backward(u.detach(), params)
        ctx.backend = backend
        ctx.phys_dtype = phys.dtype
        return res

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        u, params = ctx.saved_tensors
        be = ctx.backend
        g = (g * interior_mask(g, be.spec)).contiguous()
        gu = be.adjoint(u, g, params) if ctx.needs_input_grad[0] else None
        gp = be.param_grad(u, g, params).to(ctx.phys_dtype) if ctx.needs_input_grad[1] else None
        return gu, gp, None, None, None


def fd_residual(u: torch.Tensor, phys: torch.Tensor, dt: float, dx: float, backend) -> torch.Tensor:
    """Discrete PDE residual on the (nt, nx) grid; zeros outside the interior."""
    if not torch.is_tensor(phys):
        phys = torch.tensor([float(phys)], dtype=u.dtype, device=u.device)
    return _FDResidual.apply(u, phys.reshape(-1), float(dt), float(dx), backend)
