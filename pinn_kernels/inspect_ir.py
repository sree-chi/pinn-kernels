"""Dump the compiler's view of a residual kernel.

  python -m pinn_kernels.inspect_ir --pde heat --emit ir --opt 0     # unoptimized LLVM IR
  python -m pinn_kernels.inspect_ir --pde heat --emit ir --opt 3     # after the O3 pipeline
  python -m pinn_kernels.inspect_ir --pde burgers --emit ptx --sm sm_80
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ._jit import JitKernels, emit_ptx


def summarize_ir(ir: str) -> dict:
    vec = sorted(set(re.findall(r"<(\d+) x float>", ir)), key=int)
    return {
        "vector_widths": [int(v) for v in vec],
        "fma": len(re.findall(r"llvm\.fmuladd|llvm\.fma", ir)),
        "fdiv": len(re.findall(r"\bfdiv\b", ir)),
        "loads": len(re.findall(r"= load ", ir)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pde", default="heat", choices=["heat", "burgers"])
    p.add_argument("--emit", default="ir", choices=["ir", "ptx", "expr"])
    p.add_argument("--opt", type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument("--fast-math", action="store_true")
    p.add_argument("--sm", default="sm_75")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    if a.emit == "ptx":
        text = emit_ptx(a.pde, a.sm, fast=a.fast_math, opt=a.opt)
        entries = re.findall(r"\.entry (\w+)", text)
        print(f"# PTX for {a.pde} ({a.sm}): entries {entries}")
    else:
        k = JitKernels(a.pde, opt=a.opt, fast=a.fast_math)
        if a.emit == "expr":
            text = k.expr
        else:
            text = k.ir(optimized=True)
            print(f"# {a.pde} O{a.opt}{' fast-math' if a.fast_math else ''}: {summarize_ir(text)}")
    if a.out:
        a.out.write_text(text)
        print(f"wrote {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
