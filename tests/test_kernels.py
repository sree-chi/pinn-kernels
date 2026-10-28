"""Every kernel backend must agree with the PyTorch reference (forward,
adjoint/VJP, and parameter gradient) to float32 rounding."""

import pytest
import torch

from pinn_kernels.residual import SPECS, TorchBackend, fd_residual, get_backend, interior_mask

RTOL = 2e-6  # relative to the max magnitude of the reference


def _case(pde, nt=37, nx=53, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(nt, nx, generator=g)
    up = torch.randn(nt, nx, generator=g) * interior_mask(u, SPECS[pde])
    params = torch.tensor([0.02, 0.03, 0.05])
    return u, up, params


def _close(a, b):
    return (a - b).abs().max().item() <= RTOL * b.abs().max().item()


def _check(backend, pde):
    u, g, p = _case(pde)
    ref = TorchBackend(pde)
    assert _close(backend.forward(u, p), ref.forward(u, p))
    assert _close(backend.adjoint(u, g, p), ref.adjoint(u, g, p))
    assert _close(backend.param_grad(u, g, p), ref.param_grad(u, g, p))


@pytest.mark.parametrize("pde", ["heat", "burgers"])
@pytest.mark.parametrize("opt,fast", [(0, False), (1, False), (2, False), (3, False), (3, True)])
def test_llvm_cpu_matches_reference(pde, opt, fast):
    _check(get_backend("llvm", pde, opt=opt, fast=fast), pde)


def test_llvm_O0_forward_is_bitwise_identical():
    u, _, p = _case("heat")
    assert torch.equal(get_backend("llvm", "heat", opt=0).forward(u, p), TorchBackend("heat").forward(u, p))


@pytest.mark.parametrize("shape", [(2, 3), (3, 3), (5, 129), (130, 7)])
def test_llvm_edge_shapes(shape):
    nt, nx = shape
    u, g, p = _case("heat", nt, nx)
    b, ref = get_backend("llvm", "heat"), TorchBackend("heat")
    assert _close(b.forward(u, p), ref.forward(u, p))
    assert _close(b.adjoint(u, g, p), ref.adjoint(u, g, p))


def test_triton_matches_reference():
    pytest.importorskip("triton")
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # CPU uses TRITON_INTERPRET=1 (conftest)
    u, g, p = (t.to(dev) for t in _case("heat"))
    b, ref = get_backend("triton", "heat"), TorchBackend("heat")
    assert _close(b.forward(u, p), ref.forward(u, p))
    assert _close(b.adjoint(u, g, p), ref.adjoint(u, g, p))


@pytest.mark.gpu
@pytest.mark.parametrize("pde", ["heat", "burgers"])
def test_ptx_matches_reference(pde):
    pytest.importorskip("cupy")
    u, g, p = (t.cuda() for t in _case(pde))
    b, ref = get_backend("ptx", pde), TorchBackend(pde)
    assert _close(b.forward(u, p), ref.forward(u, p))
    assert _close(b.adjoint(u, g, p), ref.adjoint(u, g, p))


@pytest.mark.parametrize("pde", ["heat", "burgers"])
def test_autograd_function_gradcheck_float64(pde):
    # Exercises the autograd.Function wiring (masking, param grads) in float64.
    torch.manual_seed(0)
    u = torch.rand(6, 7, dtype=torch.float64, requires_grad=True)
    phys = torch.tensor([0.05], dtype=torch.float64, requires_grad=True)
    be = get_backend("torch", pde)
    assert torch.autograd.gradcheck(lambda u, a: fd_residual(u, a, 0.1, 0.2, be), (u, phys))


@pytest.mark.parametrize("pde", ["heat", "burgers"])
def test_llvm_end_to_end_gradients_match_reference(pde):
    torch.manual_seed(1)
    u0 = torch.rand(20, 30)
    grads = {}
    for name in ["torch", "llvm"]:
        u = u0.clone().requires_grad_(True)
        a = torch.tensor([0.05], requires_grad=True)
        r = fd_residual(u, a, 0.02, 0.03, get_backend(name, pde))
        (r ** 2).mean().backward()
        grads[name] = (u.grad, a.grad)
    assert _close(grads["llvm"][0], grads["torch"][0])
    assert _close(grads["llvm"][1], grads["torch"][1])
