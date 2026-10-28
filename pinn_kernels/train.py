"""Train a heat-equation PINN (forward or inverse problem).

Examples
--------
  python -m pinn_kernels.train --problem forward --physics autograd
  python -m pinn_kernels.train --problem forward --physics fd --backend llvm
  python -m pinn_kernels.train --problem inverse --physics fd --backend triton --device cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .losses import GridSampler, bc_loss, data_loss, ic_loss, physics_autograd, physics_fd
from .models import HeatPINN
from .residual import get_backend
from .sim import Domain, sensor_data, solve_heat


@dataclass
class Config:
    problem: str = "forward"         # forward | inverse
    physics: str = "fd"              # autograd | fd
    backend: str = "llvm"            # torch | llvm | triton | ptx   (fd only)
    opt: int = 3                     # LLVM optimization level (llvm backend)
    fast_math: bool = False
    epochs: int = 5000
    lr: float = 1e-3
    n_colloc: int = 5000             # autograd collocation points per step
    grid: tuple[int, int] = (64, 80)  # (nt, nx) for the fd physics loss (5120 points)
    jitter: bool = True
    width: int = 64
    depth: int = 2
    alpha_true: float = 0.05
    alpha_init: float = 0.01
    noise: float = 0.10
    inverse_ic_bc: bool = True       # the original notebook omitted these -> alpha drifts
    n_ic: int = 500
    n_bc: int = 500
    device: str = "cpu"
    seed: int = 0
    log_every: int = 500
    threads: int = 0                 # torch CPU threads (0 = default)


@dataclass
class Result:
    config: dict
    history: list = field(default_factory=list)
    rel_l2: float = float("nan")
    alpha: float | None = None
    alpha_abs_err: float | None = None
    step_ms_median: float = float("nan")
    wall_s: float = float("nan")


def evaluate(model, sol, device, n: int = 101) -> float:
    t = np.linspace(0, sol.t[-1], n)
    x = np.linspace(0, sol.x[-1], n)
    TT, XX = np.meshgrid(t, x, indexing="ij")
    u_true = sol.at(TT.ravel(), XX.ravel())
    with torch.no_grad():
        xt = torch.tensor(XX.ravel(), dtype=torch.float32, device=device).view(-1, 1)
        tt = torch.tensor(TT.ravel(), dtype=torch.float32, device=device).view(-1, 1)
        u_pred = model(xt, tt).cpu().numpy().ravel()
    return float(np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def train(cfg: Config, verbose: bool = True) -> tuple[Result, HeatPINN]:
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    dom = Domain()
    sol = solve_heat(alpha=cfg.alpha_true, nx=201, domain=dom)

    inverse = cfg.problem == "inverse"
    model = HeatPINN(cfg.width, cfg.depth, learn_alpha=inverse, alpha_init=cfg.alpha_init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if inverse:
        xs, ts, us = (a.to(device) for a in sensor_data(sol, noise=cfg.noise, seed=cfg.seed))

    backend = grid = None
    if cfg.physics == "fd":
        kw = {"opt": cfg.opt, "fast": cfg.fast_math} if cfg.backend == "llvm" else (
            {"fast": cfg.fast_math} if cfg.backend == "ptx" else {})
        backend = get_backend(cfg.backend, "heat", **kw)
        grid = GridSampler(*cfg.grid, dom.L, dom.T, device, jitter=cfg.jitter)
    fixed_alpha = torch.tensor([cfg.alpha_true], device=device)

    res = Result(config=asdict(cfg))
    step_times = []
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        _sync(device)
        ts0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        alpha = model.alpha if inverse else fixed_alpha
        if cfg.physics == "autograd":
            lp = physics_autograd(model, cfg.n_colloc, dom.L, dom.T, alpha, gen, device)
        else:
            lp = physics_fd(model, grid, alpha, backend, gen)
        terms = {"physics": lp}
        if not inverse or cfg.inverse_ic_bc:
            terms["ic"] = ic_loss(model, cfg.n_ic, dom.L, gen, device)
            terms["bc"] = bc_loss(model, cfg.n_bc, dom.L, dom.T, gen, device)
        if inverse:
            terms["data"] = data_loss(model, xs, ts, us)
        loss = sum(terms.values())
        loss.backward()
        opt.step()
        _sync(device)
        if epoch >= 10:
            step_times.append(time.perf_counter() - ts0)

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            row = {"epoch": epoch, "total": loss.item(), **{k: v.item() for k, v in terms.items()}}
            if inverse:
                row["alpha"] = model.alpha.item()
            res.history.append(row)
            if verbose:
                print(" | ".join(f"{k} {v:.5g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
                      flush=True)

    res.wall_s = time.perf_counter() - t0
    res.step_ms_median = 1e3 * statistics.median(step_times) if step_times else float("nan")
    model.eval()
    res.rel_l2 = evaluate(model, sol, device)
    if inverse:
        res.alpha = model.alpha.item()
        res.alpha_abs_err = abs(res.alpha - cfg.alpha_true)
    return res, model


def _parse() -> tuple[Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    p.add_argument("--problem", choices=["forward", "inverse"], default=d.problem)
    p.add_argument("--physics", choices=["autograd", "fd"], default=d.physics)
    p.add_argument("--backend", choices=["torch", "llvm", "triton", "ptx"], default=d.backend)
    p.add_argument("--opt", type=int, default=d.opt, choices=[0, 1, 2, 3])
    p.add_argument("--fast-math", action="store_true")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--n-colloc", type=int, default=d.n_colloc)
    p.add_argument("--grid", default="x".join(map(str, d.grid)), help="NTxNX, e.g. 64x80")
    p.add_argument("--no-jitter", action="store_true")
    p.add_argument("--no-inverse-ic-bc", action="store_true", help="reproduce the original notebook's inverse setup")
    p.add_argument("--noise", type=float, default=d.noise)
    p.add_argument("--device", default=d.device)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", type=Path, help="write the result JSON here")
    p.add_argument("--plot", type=Path, help="write a PINN-vs-FDM figure here (png)")
    a = p.parse_args()
    nt, nx = (int(v) for v in a.grid.lower().split("x"))
    cfg = Config(problem=a.problem, physics=a.physics, backend=a.backend, opt=a.opt, fast_math=a.fast_math,
                 epochs=a.epochs, lr=a.lr, n_colloc=a.n_colloc, grid=(nt, nx), jitter=not a.no_jitter,
                 inverse_ic_bc=not a.no_inverse_ic_bc, noise=a.noise, device=a.device, seed=a.seed,
                 log_every=a.log_every, threads=a.threads)
    return cfg, a


def plot(model, path: Path, alpha_true: float, device) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sol = solve_heat(alpha=alpha_true)
    x = torch.tensor(sol.x, dtype=torch.float32, device=device).view(-1, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for tv, c in zip([0.0, 0.1, 0.5, 1.0], ["C0", "C1", "C2", "C3"]):
        with torch.no_grad():
            up = model(x, torch.full_like(x, tv)).cpu().numpy().ravel()
        ax.plot(sol.x, sol.at(np.full_like(sol.x, tv), sol.x), color=c, lw=2.5, label=f"FDM t={tv}")
        ax.plot(sol.x, up, color=c, ls="--", lw=2, label=f"PINN t={tv}")
    ax.set_xlabel("x"); ax.set_ylabel("u"); ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130)


def main() -> None:
    cfg, a = _parse()
    res, model = train(cfg)
    summary = {"rel_l2": res.rel_l2, "step_ms_median": res.step_ms_median, "wall_s": res.wall_s}
    if res.alpha is not None:
        summary.update(alpha=res.alpha, alpha_abs_err=res.alpha_abs_err)
    print(json.dumps(summary, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(asdict(res), indent=2))
    if a.plot:
        plot(model, a.plot, cfg.alpha_true, torch.device(cfg.device))


if __name__ == "__main__":
    main()
