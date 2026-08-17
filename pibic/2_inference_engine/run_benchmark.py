#!/usr/bin/env python3
"""
Caso 2 — escalabilidade da verificação de GEMM.

QUATRO DEFEITOS CORRIGIDOS. A versão anterior produzia números que não mediam
o que o artigo afirmava medir:

  1. **O tamanho da matriz nunca variava.** O script passava `-DDIM_LIMIT=<n>`,
     mas `matmul_kernel.cpp` não usava o macro — M, N e K eram `2` fixos. Todos
     os cinco "tamanhos" verificavam a mesma matriz 2×2. A curva de
     escalabilidade não media escala.
  2. **`--smtlib` impede o ESBMC de resolver** — ele emite a fórmula SMT em vez
     de decidir. O critério `"VERIFICATION SUCCESSFUL" in stdout` não podia ser
     satisfeito.
  3. **Extensão `.cpp` num arquivo que é C puro**, o que fazia o frontend C++ da
     6.8.0 abortar com CONVERSION ERROR.
  4. Caminho fixo para `build/src/esbmc/esbmc`, inexistente.

O CSV anterior tinha 2 das 5 linhas (`4.5900` e `53.2923`), e o artigo publica
`<1 s` e `≈2 s` para as mesmas — 9× e 27× de distância de um dado que, além de
tudo, media a matriz errada.

Uso:
    python3 run_benchmark.py [--max-n 6] [--timeout 300]
"""

import argparse
import csv
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PIBIC = os.path.dirname(HERE)
sys.path.insert(0, PIBIC)

from core_verify.esbmc_caller import (  # noqa: E402
    ESBMC_BIN, SAFE, TIMEOUT, UNSAFE, run_esbmc,
)

KERNEL = os.path.join(HERE, "matmul_kernel.c")
OUT_CSV = os.path.join(PIBIC, "results", "case2_benchmark.csv")
OUT_MD = os.path.join(PIBIC, "results", "case2_ambiente.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()

    if ESBMC_BIN is None:
        sys.exit("ESBMC não encontrado — defina $ESBMC_BIN")

    versao = subprocess.run([ESBMC_BIN, "--version"], capture_output=True,
                            text=True).stdout.strip().splitlines()[0]
    flags = dict(memory_leak_check=True, overflow_check=True, unwind=10,
                 no_unwinding_assertions=True, boolector=True)

    print(f"ESBMC: {versao}\nkernel: {KERNEL}\n")
    linhas = []
    for n in range(2, a.max_n + 1):
        print(f"  N={n} ...", end=" ", flush=True)
        r = run_esbmc(KERNEL, timeout=a.timeout,
                      extra_args=[f"-DDIM_LIMIT={n}"], **flags)
        linhas.append((n, r.time_taken, r.status))
        print(f"{r.status} em {r.time_taken:.2f}s")
        if r.status == TIMEOUT:
            print(f"  (parando: N={n} nao decide em {a.timeout}s)")
            break

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["MatrixSize", "Time(s)", "Status"])
        for n, t, s in linhas:
            w.writerow([n, f"{t:.4f}", s])

    # o ambiente vai junto: sem ele o numero nao e comparavel (KB-D09)
    with open(OUT_MD, "w") as fh:
        fh.write(f"""# Caso 2 — ambiente da medição

Gerado por `2_inference_engine/run_benchmark.py`. Não editar à mão.

| | |
|---|---|
| ESBMC | {versao} |
| Solver | Boolector (padrão) |
| Flags | `--memory-leak-check --overflow-check --unwind 10 --no-unwinding-assertions -DDIM_LIMIT=N` |
| Timeout | {a.timeout} s |
| Kernel | `2_inference_engine/matmul_kernel.c`, M=N=K=TILE=`DIM_LIMIT` |
| SO | {platform.platform()} |
| CPU | {os.cpu_count()} núcleos |

| N | Tempo (s) | Status |
|---|---|---|
""")
        for n, t, s in linhas:
            fh.write(f"| {n} | {t:.2f} | {s} |\n")
        fh.write("\n`TIMEOUT` é **indeciso**, não “seguro”.\n")

    print(f"\n{OUT_CSV}\n{OUT_MD}")


if __name__ == "__main__":
    main()
