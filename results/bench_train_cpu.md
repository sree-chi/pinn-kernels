Device: x86_64 | torch 2.14.1+cu130 | threads 1

| physics | points | step_ms | rel_l2 |
|---|---|---|---|
| autograd | 5000 | 30.305 | 0.225 |
| fd/torch | 5120 | 7.174 | 0.218 |
| fd/llvm | 5120 | 6.633 | 0.218 |
