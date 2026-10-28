# pinn-kernels

Physics-informed neural networks (PINNs) for the 1D heat equation, where the
physics loss runs on **compiler-generated kernels**. A PDE residual is written
once as a stencil expression graph. An LLVM-based code generator turns it into:
- vectorized CPU kernels (JIT-compiled through ORC);
- GPU kernels, emitted as PTX by LLVM's NVPTX backend;
- the matching **adjoint (backward) kernels**, derived by symbolically differentiating the graph.

Hand-written Triton kernels provide a second GPU path. All backends plug into
PyTorch autograd, so the network trains through them end to end.

```
                 stencil expression graph (csrc/graph.h)
                 r = (u[t+1,x]-u[t,x])/dt - alpha*(u[t,x+1]-2u[t,x]+u[t,x-1])/dx^2
                        │                         │
           symbolic d/du, d/dalpha          IRBuilder codegen (csrc/codegen.cpp)
                        │            ┌────────────┴─────────────┐
                        ▼            ▼                          ▼
              adjoint + param-grad   CPU: O0..O3 pass pipeline   GPU: NVPTX backend → PTX
              kernels                → ORC LLJIT (AVX2 vectors)  → loaded with CuPy
                        └────────────┬──────────────────────────┘
                                     ▼
   MLP u(x,t) on an (nt,nx) grid ──► torch.autograd.Function ──► loss.backward()
                                     (forward kernel / adjoint kernel)
```

## Results (measured)

CPU numbers come from a 2-core cloud VM with an x86 AVX2 CPU, single thread.
**GPU numbers: run `notebooks/01_colab_gpu.ipynb`**; none are filled in here yet.

**Inverse problem: recover thermal diffusivity alpha (true 0.05) from 3 noisy sensors** (10% noise, 10,000 Adam steps, `results/inverse_*.json`):

| physics loss | alpha found | abs error | rel. L2 vs FDM | step time |
|---|---|---|---|---|
| autograd, 5,000 random points (original method) | 0.050456 | 4.6e-4 | 0.54% | 19.1 ms |
| finite-difference grid 64×80, LLVM kernels | 0.049997 | 2.7e-6 | 1.18% | 5.3 ms |
| FD + LLVM, **no IC/BC loss** (original notebook setup) | 0.048850 | 1.1e-3 | 13.9% | 3.9 ms |

**Residual kernel, heat equation, CPU single thread** (`results/bench_kernels_cpu.md`). Bandwidth assumes 8 bytes moved per grid point; `copy` is `tensor.clone()`, the practical ceiling.

| grid | backend | forward | adjoint | forward GB/s |
|---|---|---|---|---|
| 2048² | copy (ceiling) | 1.65 ms | | 20.3 |
| 2048² | PyTorch eager | 34.0 ms | 24.3 ms | 1.0 |
| 2048² | torch.compile (Inductor) | 5.32 ms | 12.2 ms | 6.3 |
| 2048² | LLVM O0 | 17.9 ms | 29.0 ms | 1.9 |
| 2048² | **LLVM O3** | **2.82 ms** | **5.14 ms** | **11.9** |

**Training step, forward problem, CPU single thread** (`results/bench_train_cpu.md`): autograd 30.3 ms, FD/PyTorch 7.2 ms, FD/LLVM 6.6 ms.

What these numbers say, plainly:
- **The big win is the method, not the kernel.**
  - The finite-difference physics loss avoids second-order backprop through the MLP. That makes a training step **~4.5× faster** than autograd at a similar point count.
  - At a 64×80 grid the MLP dominates, so the custom kernel adds only ~8% over PyTorch slicing.
  - The kernels matter at large grids: there LLVM O3 is 12× faster than eager PyTorch and 1.9× faster than `torch.compile` (forward).
- **Accuracy trade-off.** Forward Euler in time has O(dt) truncation error, so the FD loss has higher field error (1.18% vs 0.54%). It still recovered alpha more precisely in this run.
- **The original inverse notebook drifted** (alpha 0.054 → 0.036) because the loss had no initial or boundary conditions, so alpha and the solution shape could trade off. Adding IC/BC terms fixes it, and the ablation row reproduces the drift.

## What's in here

| path | what |
|---|---|
| `csrc/graph.h` | Expression DAG, simplifier, symbolic differentiation, PDE library (heat; Burgers to show a nonlinear residual) |
| `csrc/codegen.cpp` | IRBuilder codegen. Forward, adjoint and parameter-gradient kernels; `noalias` args; 64-bit `nsw` indexing; new-pass-manager O0–O3; ORC LLJIT; NVPTX/PTX emission; C ABI |
| `pinn_kernels/residual.py` | Backends (`torch`, `llvm`, `triton`, `ptx`) plus the `autograd.Function` bridge |
| `pinn_kernels/triton_kernels.py` | Triton forward and adjoint kernels (heat), with parameters read on-device |
| `pinn_kernels/{sim,models,losses,train}.py` | Vectorized FDM ground truth, PINN, losses, training CLI |
| `pinn_kernels/bench.py` | Kernel bandwidth benchmark (with a Triton block-size sweep on GPU) and training-step benchmark |
| `pinn_kernels/inspect_ir.py` | Dump O0/O3 LLVM IR or PTX and summarize vector widths |
| `tests/` | 39 tests: backend parity (forward, adjoint, param grad) vs PyTorch autograd, gradcheck, codegen properties, PTX emission, training smoke tests |
| `notebooks/00_original_exploration.ipynb` | The original notebook this grew from |
| `notebooks/01_colab_gpu.ipynb` | Builds everything on Colab and runs the GPU tests, benchmarks and training |

## Run it

```bash
# needs LLVM 18 dev headers (Ubuntu 24.04: apt install llvm-18-dev libzstd-dev libedit-dev)
make install            # cmake build of libpinnjit.so + pip install -e .
make test
python -m pinn_kernels.train --problem inverse --physics fd --backend llvm --plot results/inverse.png
python -m pinn_kernels.bench kernels --device cpu
python -m pinn_kernels.inspect_ir --pde heat --opt 3        # see the <8 x float> loop
python -m pinn_kernels.inspect_ir --pde heat --emit ptx --sm sm_80
```

Docker:

```bash
make docker-cpu && docker run --rm pinn-kernels:cpu                      # tests + CPU kernel bench
make docker-gpu && docker run --gpus all --rm pinn-kernels:gpu make bench-gpu
```

CI (GitHub Actions) builds the code generator, runs the test suite (Triton kernels run under
`TRITON_INTERPRET=1`), uploads the generated IR and PTX as artifacts, and builds and tests the CPU Docker image.

## How the adjoint is generated

For a residual R that depends on stencil loads u[t+a, x+b], the VJP is

```
gu[t, x] = Σ_(a,b)  ∂R/∂u[·+a, ·+b] evaluated at (t-a, x-b)  ·  g[t-a, x-b]
```

- The code generator gets each ∂R/∂u term by differentiating the graph.
- It emits the sum as a gather over g, which is zero-padded by the stencil halo, so there's no branching and no write conflicts. The same form works on CPU and GPU.
- For nonlinear residuals (Burgers), the coefficients depend on u. Those loads are clamped into the grid; wherever g is zero, the clamped value has no effect.
- Parameter gradients (dL/dalpha) come from reduction kernels generated from ∂R/∂alpha.
- Every path is tested against PyTorch autograd.

## Limitations / next steps

- **CPU kernels are single-threaded.** Next step: row-parallel outer loop.
- **Triton kernels cover only the heat equation.** The LLVM path handles any graph.
- **The PTX path has no parameter-gradient kernel yet.** dL/dalpha uses PyTorch there.
- **`fast` mode doesn't rewrite divisions** by loop-invariant parameters into reciprocal multiplies. A graph-level rewrite would do it.
- **MLIR is used only indirectly** (Triton lowers through MLIR → LLVM IR → PTX). Expressing the stencil in MLIR's `linalg`/`affine` dialects would be the natural extension.
