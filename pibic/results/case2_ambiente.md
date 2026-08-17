# Caso 2 — ambiente da medição

Gerado por `2_inference_engine/run_benchmark.py`. Não editar à mão.

| | |
|---|---|
| ESBMC | ESBMC version 6.8.0 64-bit x86_64 linux |
| Solver | Boolector (padrão) |
| Flags | `--memory-leak-check --overflow-check --unwind 10 --no-unwinding-assertions -DDIM_LIMIT=N` |
| Timeout | 300 s |
| Kernel | `2_inference_engine/matmul_kernel.c`, M=N=K=TILE=`DIM_LIMIT` |
| SO | Linux-6.18.5-fc-v20-x86_64-with-glibc2.39 |
| CPU | 4 núcleos |

| N | Tempo (s) | Status |
|---|---|---|
| 2 | 1.97 | SAFE |
| 3 | 16.83 | SAFE |
| 4 | 300.05 | TIMEOUT |

`TIMEOUT` é **indeciso**, não “seguro”.
