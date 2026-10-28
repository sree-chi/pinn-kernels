"""Properties of the generated code itself."""

import re

import pytest

from pinn_kernels._jit import JitKernels, emit_ptx


def test_O3_vectorizes_and_O0_does_not():
    o0 = JitKernels("heat", opt=0).ir()
    o3 = JitKernels("heat", opt=3).ir()
    assert not re.search(r"<\d+ x float>", o0)
    assert re.search(r"<\d+ x float>", o3), "loop vectorizer did not fire"


def test_noalias_on_kernel_pointers():
    ir = JitKernels("heat", opt=0).ir(optimized=False)
    sig = next(line for line in ir.splitlines() if line.startswith("define") and "@pk_fwd" in line)
    assert sig.count("noalias") == 3


def test_kernels_and_params_are_derived_from_the_graph():
    k = JitKernels("burgers")
    assert k.params == ["dt", "dx", "nu"] and k.physics_params == ["nu"]
    assert (k.t_lo, k.t_hi, k.x_lo, k.x_hi, k.halo) == (0, 1, 1, 1, 1)
    ir = k.ir()
    for fn in ["@pk_fwd", "@pk_adj", "@pk_pgrad_nu"]:
        assert fn in ir


@pytest.mark.parametrize("pde", ["heat", "burgers"])
@pytest.mark.parametrize("sm", ["sm_75", "sm_80", "sm_90"])
def test_ptx_emission(pde, sm):
    ptx = emit_ptx(pde, sm)
    assert f".target {sm}" in ptx
    assert re.findall(r"\.visible \.entry (\w+)", ptx) == ["pk_fwd", "pk_adj"]
    assert "ld.global" in ptx and "st.global" in ptx


def test_unknown_pde_raises():
    with pytest.raises(RuntimeError, match="unknown PDE"):
        JitKernels("wave")
