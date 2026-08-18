# Evidências de execução

Gerado por `scripts/gerar_evidencias.py`. **Não editar à mão** — reexecute.

Cada linha aponta para a saída **bruta** do solver em `results/logs/`. Existe
porque `KB-D03` registrou que nenhum log versionado continha um veredito,
enquanto o capítulo 4 afirmava provas bem-sucedidas.

`TIMEOUT` é **indeciso**: não significa seguro nem inseguro.

| | |
|---|---|
| ESBMC | ESBMC version 6.8.0 64-bit x86_64 linux |
| SO | Linux-6.18.5-fc-v20-x86_64-with-glibc2.39 |
| CPU | 4 núcleos |
| Duração total | 463 s |

| Caso | Harness | Veredito | Tempo | Comando | Log |
|---|---|---|---:|---|---|
| Caso 1 | MLP quantizada (XOR) | **SAFE** | 0.08 s | `--no-unwinding-assertions` | [`verify_mlp_qnn.log`](logs/verify_mlp_qnn.log) |
| Caso 1 | MLP stub (float) | **SAFE** | 3.09 s | `--floatbv --no-unwinding-assertions` | [`mlp_stub.log`](logs/mlp_stub.log) |
| Caso 1 | Rede + propriedade | **UNSAFE** | 197.82 s | `--floatbv --no-unwinding-assertions` | [`property_check.log`](logs/property_check.log) |
| Caso 1 | Transformer (DeepSeek stub) | **SAFE** | 0.08 s | `--floatbv --overflow-check --no-unwinding-assertions --unwind 4` | [`deepseek_mlp_stub.log`](logs/deepseek_mlp_stub.log) |
| Caso 2 | Kernels de inferência | **UNSAFE** | 0.43 s | `--memory-leak-check --overflow-check --no-pointer-check --no-unwinding-assertions --z3 --unwind 4` | [`kernels_benchmarks.log`](logs/kernels_benchmarks.log) |
| Caso 2 | GEMM com tiling (N=3) | **SAFE** | 17.51 s | `--memory-leak-check --overflow-check --no-unwinding-assertions --boolector --unwind 10 -DDIM_LIMIT=3` | [`matmul_kernel.log`](logs/matmul_kernel.log) |
| Caso 4 | PID sob ruído [-5,+5], 10 passos | **SAFE** | 213.47 s | `--floatbv --no-unwinding-assertions --unwind 11` | [`pid_controller.log`](logs/pid_controller.log) |
| Caso 6 | Política RL (bounds do atuador) | **UNSAFE** | 2.58 s | `--floatbv --z3 --unwind 1` | [`rl_policy.log`](logs/rl_policy.log) |
| FFN/LLM | GPT-2 2x4 (QNN, GeLU exato) | **SAFE** | 0.24 s | `--boolector --no-unwinding-assertions` | [`gpt2_2x4_qnn.log`](logs/gpt2_2x4_qnn.log) |
| FFN/LLM | GPT-2 4x8 (QNN, GeLU exato) | **SAFE** | 4.24 s | `--boolector --no-unwinding-assertions` | [`gpt2_4x8_qnn.log`](logs/gpt2_4x8_qnn.log) |
| FFN/LLM | LLaMA-7B 4x8 (QNN, SiLU exato) | **SAFE** | 3.46 s | `--boolector --no-unwinding-assertions` | [`llama-7b_4x8_qnn.log`](logs/llama-7b_4x8_qnn.log) |
| FFN/LLM | GPT-2 4x16 (QNN, GeLU exato) | **SAFE** | 20.10 s | `--boolector --no-unwinding-assertions` | [`gpt2_4x16_qnn.log`](logs/gpt2_4x16_qnn.log) |

**9 provados · 3 com contraexemplo · 0 indecisos** de 12 harnesses.
