"""ctypes bindings to the LLVM code generator (csrc/codegen.cpp -> libpinnjit.so)."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

_lib = None


def _find_lib() -> Path:
    env = os.environ.get("PINNJIT_LIB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "_lib" / "libpinnjit.so"


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    path = _find_lib()
    if not path.exists():
        raise RuntimeError(f"{path} not found. Build the LLVM library first: `make build` "
                           "(needs LLVM 18 dev headers), or set PINNJIT_LIB.")
    L = ctypes.CDLL(str(path))
    vp, i64, f32p = ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p
    sig = {
        "pk_last_error": (ctypes.c_char_p, []),
        "pk_compile": (vp, [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]),
        "pk_free": (None, [vp]),
        "pk_num_params": (ctypes.c_int, [vp]),
        "pk_param_name": (ctypes.c_char_p, [vp, ctypes.c_int]),
        "pk_num_physics_params": (ctypes.c_int, [vp]),
        "pk_physics_param_name": (ctypes.c_char_p, [vp, ctypes.c_int]),
        "pk_bounds": (None, [vp, ctypes.POINTER(ctypes.c_int64)]),
        "pk_forward": (None, [vp, f32p, f32p, f32p, i64, i64]),
        "pk_adjoint": (None, [vp, f32p, f32p, f32p, f32p, i64, i64]),
        "pk_param_grad": (ctypes.c_double, [vp, ctypes.c_int, f32p, f32p, f32p, i64, i64]),
        "pk_ir": (ctypes.c_char_p, [vp, ctypes.c_int]),
        "pk_residual_expr": (ctypes.c_char_p, [vp]),
        "pk_emit_ptx": (ctypes.c_char_p, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(L, name)
        fn.restype, fn.argtypes = res, args
    _lib = L
    return L


def _ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _check_cpu_f32(*ts: torch.Tensor) -> None:
    for t in ts:
        if t.device.type != "cpu" or t.dtype != torch.float32 or not t.is_contiguous():
            raise ValueError("LLVM CPU kernels need contiguous float32 CPU tensors")


class JitKernels:
    """JIT-compiled forward / adjoint / parameter-gradient kernels for one PDE."""

    def __init__(self, pde: str = "heat", opt: int = 3, fast: bool = False):
        L = lib()
        self._h = L.pk_compile(pde.encode(), int(opt), int(fast))
        if not self._h:
            raise RuntimeError(L.pk_last_error().decode())
        self.pde, self.opt, self.fast = pde, opt, fast
        self.params = [L.pk_param_name(self._h, i).decode() for i in range(L.pk_num_params(self._h))]
        self.physics_params = [L.pk_physics_param_name(self._h, i).decode()
                               for i in range(L.pk_num_physics_params(self._h))]
        b = (ctypes.c_int64 * 5)()
        L.pk_bounds(self._h, b)
        self.t_lo, self.t_hi, self.x_lo, self.x_hi, self.halo = list(b)

    def __del__(self):
        if getattr(self, "_h", None) and _lib is not None:
            _lib.pk_free(self._h)
            self._h = None

    @property
    def expr(self) -> str:
        return lib().pk_residual_expr(self._h).decode()

    def ir(self, optimized: bool = True) -> str:
        return lib().pk_ir(self._h, int(optimized)).decode()

    def forward(self, u: torch.Tensor, params: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        _check_cpu_f32(u, params)
        nt, nx = u.shape
        if out is None:
            out = torch.zeros_like(u)  # boundary rows/cols stay 0
        lib().pk_forward(self._h, _ptr(u), _ptr(out), _ptr(params), nt, nx)
        return out

    def adjoint(self, gpad: torch.Tensor, u: torch.Tensor, params: torch.Tensor,
                out: torch.Tensor | None = None) -> torch.Tensor:
        _check_cpu_f32(gpad, u, params)
        nt, nx = u.shape
        assert gpad.shape == (nt + 2 * self.halo, nx + 2 * self.halo)
        if out is None:
            out = torch.empty_like(u)
        lib().pk_adjoint(self._h, _ptr(gpad), _ptr(out), _ptr(u), _ptr(params), nt, nx)
        return out

    def param_grad(self, k: int, u: torch.Tensor, g: torch.Tensor, params: torch.Tensor) -> float:
        _check_cpu_f32(u, g, params)
        nt, nx = u.shape
        return lib().pk_param_grad(self._h, k, _ptr(u), _ptr(g), _ptr(params), nt, nx)


def emit_ptx(pde: str = "heat", sm: str = "sm_75", fast: bool = False, opt: int = 3) -> str:
    L = lib()
    out = L.pk_emit_ptx(pde.encode(), sm.encode(), int(fast), int(opt))
    if not out:
        raise RuntimeError(L.pk_last_error().decode())
    return out.decode()
