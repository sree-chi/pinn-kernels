"""Benchmarks.

  python -m pinn_kernels.bench kernels --device cpu   [--sizes 512,1024,2048]
  python -m pinn_kernels.bench kernels --device cuda
  python -m pinn_kernels.bench train   --device cpu   [--epochs 300]

`kernels` times the heat residual forward and adjoint kernels per backend and
reports effective memory bandwidth (8 bytes moved per grid point: one float
read, one float written, assuming perfect stencil reuse) next to a plain
tensor copy, which is the practical bandwidth ceiling for a memory-bound
stencil. `train` measures median optimizer-step time for each physics path.

CPU note: the LLVM kernels are single-threaded, so CPU numbers are taken with
torch.set_num_threads(1) unless --threads is given.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .residual import LLVMBackend, TorchBackend, reference_residual


def _timer(fn, device, reps: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); b.synchronize()
            ts.append(a.elapsed_time(b) / 1e3)
        return statistics.median(ts)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def heat_adjoint_explicit(g: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Closed-form heat adjoint in plain PyTorch (eager / torch.compile baseline)."""
    dt, dx, a = params[0], params[1], params[2]
    k = a / (dx * dx)
    gp = F.pad(g, (1, 1, 1, 1))
    c, up, lft, rgt = gp[1:-1, 1:-1], gp[:-2, 1:-1], gp[1:-1, :-2], gp[1:-1, 2:]
    return (2.0 * k - 1.0 / dt) * c + up / dt - k * (lft + rgt)


def kernel_backends(device: torch.device, triton_sweep: bool):
    """Yield (name, forward(u, g, p), adjoint(u, g, p)) closures."""
    yield ("torch eager", lambda u, g, p: reference_residual("heat", u, p),
           lambda u, g, p: heat_adjoint_explicit(g, p))
    try:
        cf = torch.compile(lambda u, p: reference_residual("heat", u, p), dynamic=False)
        ca = torch.compile(heat_adjoint_explicit, dynamic=False)
        yield ("torch.compile", lambda u, g, p: cf(u, p), lambda u, g, p: ca(g, p))
    except Exception as e:  # pragma: no cover
        print(f"torch.compile unavailable: {e}")

    if device.type == "cpu":
        for opt, fast in [(0, False), (3, False), (3, True)]:
            b = LLVMBackend("heat", opt=opt, fast=fast)
            yield (f"llvm O{opt}{' fast-math' if fast else ''}",
                   (lambda b: lambda u, g, p: b.forward(u, p))(b),
                   (lambda b: lambda u, g, p: b.adjoint(u, g, p))(b))
    else:
        from . import triton_kernels as tk

        configs = [(4, 64), (8, 64), (8, 128), (16, 64), (32, 32), (4, 256)] if triton_sweep else [(8, 64)]
        for bt, bx in configs:
            yield (f"triton {bt}x{bx}",
                   (lambda bt, bx: lambda u, g, p: tk.heat_forward(u, p, bt, bx))(bt, bx),
                   (lambda bt, bx: lambda u, g, p: tk.heat_adjoint(F.pad(g, (1, 1, 1, 1)), u.shape, p, bt, bx))(bt, bx))
        try:
            from .residual import PTXBackend

            b = PTXBackend("heat")
            yield ("llvm nvptx", lambda u, g, p: b.forward(u, p), lambda u, g, p: b.adjoint(u, g, p))
        except Exception as e:
            print(f"NVPTX backend skipped: {e}")


def bench_kernels(device: torch.device, sizes, reps: int, triton_sweep: bool):
    rows = []
    for n in sizes:
        u = torch.rand(n, n, device=device)
        g = torch.randn(n, n, device=device)
        p = torch.tensor([1e-3, 1e-2, 0.05], device=device)
        bytes_moved = 8 * n * n
        t_copy = _timer(lambda: u.clone(), device, reps)
        rows.append({"size": n, "backend": "copy (ceiling)", "fwd_ms": t_copy * 1e3, "adj_ms": None,
                     "fwd_GBs": bytes_moved / t_copy / 1e9, "adj_GBs": None})
        for name, fwd, adj in kernel_backends(device, triton_sweep):
            try:
                tf = _timer(lambda: fwd(u, g, p), device, reps)
                ta = _timer(lambda: adj(u, g, p), device, reps)
            except Exception as e:
                print(f"{name} @ {n}: failed ({e})")
                continue
            rows.append({"size": n, "backend": name, "fwd_ms": tf * 1e3, "adj_ms": ta * 1e3,
                         "fwd_GBs": bytes_moved / tf / 1e9, "adj_GBs": bytes_moved / ta / 1e9})
            print(f"{n:>5} {name:<22} fwd {tf*1e3:8.3f} ms ({bytes_moved/tf/1e9:6.1f} GB/s)  "
                  f"adj {ta*1e3:8.3f} ms ({bytes_moved/ta/1e9:6.1f} GB/s)", flush=True)
    return rows


def bench_train(device: torch.device, epochs: int):
    from .train import Config, train

    variants = [("autograd", "torch"), ("fd", "torch"), ("fd", "llvm")] if device.type == "cpu" else \
               [("autograd", "torch"), ("fd", "torch"), ("fd", "triton"), ("fd", "ptx")]
    rows = []
    for physics, backend in variants:
        cfg = Config(problem="forward", physics=physics, backend=backend, epochs=epochs,
                     device=str(device), log_every=10**9)
        try:
            res, _ = train(cfg, verbose=False)
        except Exception as e:
            print(f"{physics}/{backend}: failed ({e})")
            continue
        label = physics if physics == "autograd" else f"fd/{backend}"
        points = cfg.n_colloc if physics == "autograd" else cfg.grid[0] * cfg.grid[1]
        rows.append({"physics": label, "points": points, "step_ms": res.step_ms_median, "rel_l2": res.rel_l2})
        print(f"{label:<12} points {points:>6}  step {res.step_ms_median:7.2f} ms  rel L2 {res.rel_l2:.4f}", flush=True)
    return rows


def to_markdown(rows) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    fmt = lambda v: "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["kernels", "train"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sizes", default=None, help="comma-separated square grid sizes")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no-triton-sweep", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args()

    device = torch.device(a.device)
    if device.type == "cpu":
        torch.set_num_threads(a.threads)
    sizes = [int(s) for s in a.sizes.split(",")] if a.sizes else (
        [512, 1024, 2048] if device.type == "cpu" else [1024, 2048, 4096, 8192])

    meta = {"device": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor() or platform.machine(),
            "torch": torch.__version__, "threads": torch.get_num_threads() if device.type == "cpu" else None}
    rows = bench_kernels(device, sizes, a.reps, not a.no_triton_sweep) if a.what == "kernels" else \
        bench_train(device, a.epochs)

    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"bench_{a.what}_{device.type}"
    (a.out / f"{stem}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    (a.out / f"{stem}.md").write_text(f"Device: {meta['device']} | torch {meta['torch']}"
                                      f"{' | threads ' + str(meta['threads']) if meta['threads'] else ''}\n\n"
                                      + to_markdown(rows) + "\n")
    print(f"\nwrote {a.out / (stem + '.md')}")


if __name__ == "__main__":
    main()
