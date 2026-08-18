#!/usr/bin/env python3
"""
gerar_evidencias.py — Executa cada harness citado no artigo e grava a saída
bruta do solver, produzindo `results/EVIDENCIAS.md` com a tabela
afirmação → comando → veredito → log.

POR QUE ISTO EXISTE. `KB-D03` registra que **nenhum log em `results/` continha
um veredito**: o capítulo 4 afirmava `VERIFICATION SUCCESSFUL` em vários pontos
sem que um único artefato de execução estivesse versionado. Commitar um log
avulso resolveria o sintoma; o problema era não haver como regerar a evidência.

Regras que este runner segue, e que a versão anterior do projeto não seguia:

  - a saída **bruta** do ESBMC é sempre salva, mesmo quando nenhum marcador
    casa. "Nenhuma linha casou" e "não rodou" precisam ser distinguíveis;
  - `TIMEOUT` é registrado como **indeciso**, nunca como sucesso ou falha;
  - o ambiente (versão, solver, flags, máquina) vai junto dos números, porque
    sem ele o tempo não é comparável entre execuções;
  - harnesses que não compilam aparecem na tabela como `PARSE_ERROR`, em vez
    de sumirem silenciosamente.

Uso:
    python3 scripts/gerar_evidencias.py            # tudo
    python3 scripts/gerar_evidencias.py --rapido   # pula os > 60 s
"""

import argparse
import os
import platform
import subprocess
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
PIBIC = os.path.dirname(AQUI)
sys.path.insert(0, PIBIC)

from core_verify.esbmc_caller import ESBMC_BIN, run_esbmc  # noqa: E402

LOGS = os.path.join(PIBIC, "results", "logs")
SAIDA = os.path.join(PIBIC, "results", "EVIDENCIAS.md")

# (caso, rótulo, caminho, flags, timeout_s, lento?)
HARNESSES = [
    ("Caso 1", "MLP quantizada (XOR)", "teste_mlp/verify_mlp_qnn.c",
     dict(no_unwinding_assertions=True), 120, False),
    ("Caso 1", "MLP stub (float)", "verification/mlp_stub.c",
     dict(floatbv=True, no_unwinding_assertions=True), 120, False),
    ("Caso 1", "Rede + propriedade", "verification/property_check.c",
     dict(floatbv=True, no_unwinding_assertions=True), 240, True),
    ("Caso 1", "Transformer (DeepSeek stub)", "verification/deepseek_mlp_stub.c",
     dict(floatbv=True, unwind=4, overflow_check=True,
          no_unwinding_assertions=True), 240, True),
    ("Caso 2", "Kernels de inferência", "cases/inference_safety/kernels_benchmarks.c",
     dict(memory_leak_check=True, overflow_check=True, no_pointer_check=True,
          no_unwinding_assertions=True, unwind=4, z3=True), 120, False),
    ("Caso 2", "GEMM com tiling (N=3)", "2_inference_engine/matmul_kernel.c",
     dict(memory_leak_check=True, overflow_check=True, unwind=10,
          no_unwinding_assertions=True, boolector=True,
          extra_args=["-DDIM_LIMIT=3"]), 120, False),
    ("Caso 6", "Política RL (bounds do atuador)", "cases/ai_model_checking/rl_policy.c",
     dict(floatbv=True, unwind=1, z3=True), 120, False),
    ("FFN/LLM", "GPT-2 2x4 (QNN, GeLU exato)",
     "cases/llm_ffn_verification/verify_output/gpt2_2x4_qnn.c",
     dict(boolector=True, no_unwinding_assertions=True), 120, False),
    ("FFN/LLM", "GPT-2 4x8 (QNN, GeLU exato)",
     "cases/llm_ffn_verification/verify_output/gpt2_4x8_qnn.c",
     dict(boolector=True, no_unwinding_assertions=True), 120, False),
    ("FFN/LLM", "LLaMA-7B 4x8 (QNN, SiLU exato)",
     "cases/llm_ffn_verification/verify_output/llama-7b_4x8_qnn.c",
     dict(boolector=True, no_unwinding_assertions=True), 120, False),
    ("FFN/LLM", "GPT-2 4x16 (QNN, GeLU exato)",
     "cases/llm_ffn_verification/verify_output/gpt2_4x16_qnn.c",
     dict(boolector=True, no_unwinding_assertions=True), 120, True),
]

FLAG_CLI = {
    "floatbv": "--floatbv", "z3": "--z3", "boolector": "--boolector",
    "memory_leak_check": "--memory-leak-check", "overflow_check": "--overflow-check",
    "no_pointer_check": "--no-pointer-check",
    "no_unwinding_assertions": "--no-unwinding-assertions",
}


def cmd_legivel(flags):
    partes = [FLAG_CLI[k] for k in flags if k in FLAG_CLI and flags[k]]
    if flags.get("unwind") is not None:
        partes += ["--unwind", str(flags["unwind"])]
    partes += flags.get("extra_args", [])
    return " ".join(partes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="pula os harnesses lentos")
    a = ap.parse_args()

    if ESBMC_BIN is None:
        sys.exit("ESBMC não encontrado — defina $ESBMC_BIN")
    versao = subprocess.run([ESBMC_BIN, "--version"], capture_output=True,
                            text=True).stdout.strip().splitlines()[0]
    os.makedirs(LOGS, exist_ok=True)

    linhas, t0 = [], time.time()
    for caso, rotulo, rel, flags, tmo, lento in HARNESSES:
        if a.rapido and lento:
            linhas.append((caso, rotulo, rel, flags, "PULADO", 0.0, None))
            print(f"  {rotulo}: pulado")
            continue
        alvo = os.path.join(PIBIC, rel)
        if not os.path.isfile(alvo):
            linhas.append((caso, rotulo, rel, flags, "AUSENTE", 0.0, None))
            print(f"  {rotulo}: ARQUIVO AUSENTE")
            continue

        print(f"  {rotulo} ...", end=" ", flush=True)
        r = run_esbmc(alvo, timeout=tmo, **flags)
        nome = os.path.basename(rel).replace(".c", "") + ".log"
        with open(os.path.join(LOGS, nome), "w", encoding="utf-8") as fh:
            fh.write(f"$ esbmc {rel} {cmd_legivel(flags)}\n")
            fh.write(f"# {versao}\n# status={r.status} rc={r.returncode} "
                     f"tempo={r.time_taken:.2f}s\n\n")
            fh.write(r.output)          # SEMPRE a saida bruta, sem filtro
        linhas.append((caso, rotulo, rel, flags, r.status, r.time_taken, nome))
        print(f"{r.status} em {r.time_taken:.2f}s")

    with open(SAIDA, "w", encoding="utf-8") as fh:
        fh.write(f"""# Evidências de execução

Gerado por `scripts/gerar_evidencias.py`. **Não editar à mão** — reexecute.

Cada linha aponta para a saída **bruta** do solver em `results/logs/`. Existe
porque `KB-D03` registrou que nenhum log versionado continha um veredito,
enquanto o capítulo 4 afirmava provas bem-sucedidas.

`TIMEOUT` é **indeciso**: não significa seguro nem inseguro.

| | |
|---|---|
| ESBMC | {versao} |
| SO | {platform.platform()} |
| CPU | {os.cpu_count()} núcleos |
| Duração total | {time.time() - t0:.0f} s |

| Caso | Harness | Veredito | Tempo | Comando | Log |
|---|---|---|---:|---|---|
""")
        for caso, rot, rel, flags, st, t, log in linhas:
            ref = f"[`{log}`](logs/{log})" if log else "—"
            fh.write(f"| {caso} | {rot} | **{st}** | {t:.2f} s | "
                     f"`{cmd_legivel(flags)}` | {ref} |\n")

        ok = sum(1 for *_, st, _, _ in linhas if st == "SAFE")
        ns = sum(1 for *_, st, _, _ in linhas if st == "UNSAFE")
        ind = sum(1 for *_, st, _, _ in linhas if st in ("TIMEOUT", "UNKNOWN"))
        fh.write(f"\n**{ok} provados · {ns} com contraexemplo · {ind} indecisos** "
                 f"de {len(linhas)} harnesses.\n")

    print(f"\n{SAIDA}\n{LOGS}/")


if __name__ == "__main__":
    main()
