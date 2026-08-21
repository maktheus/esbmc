#!/usr/bin/env bash
# run_pdr.sh — Sintetiza o sistema de transição e prova a propriedade de
# segurança com IC3/PDR (ABC), reportando tempo e pico de memória reais.
#
#   ./run_pdr.sh cl_ddpg16.v [segundos]
#
# Requer: yosys e um executavel ABC (`yosys-abc` ou `abc`). Antes de rodar:
#   python3 check_dependencies.py
# YOSYS=/caminho/yosys ABC=/caminho/yosys-abc PYTHON=python3 ./run_pdr.sh ...
#
# Saída do ABC:
#   "Property proved"      -> invariante indutivo encontrado, seguro para SEMPRE
#   "Output ... asserted"  -> contraexemplo concreto no passo indicado
#   "Timeout"/"Unfinished" -> indeciso; nao confundir com "seguro"
set -euo pipefail

SRC="${1:?uso: run_pdr.sh <arquivo.v> [timeout_s]}"
TMO="${2:-1800}"

if [[ ! -f "$SRC" ]]; then
  echo "ERRO: fonte Verilog nao encontrada: $SRC" >&2
  exit 2
fi
if [[ ! "$TMO" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERRO: timeout deve ser um numero inteiro positivo (recebido: $TMO)" >&2
  exit 2
fi

find_tool() {
  local configured="$1"
  shift
  if [[ -n "$configured" ]]; then
    command -v "$configured" 2>/dev/null || return 1
    return
  fi
  local candidate
  for candidate in "$@"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done
  return 1
}

YOSYS_BIN="$(find_tool "${YOSYS:-}" yosys)" || {
  echo "DEPENDENCIA AUSENTE: Yosys. Instale-o ou defina YOSYS=/caminho/yosys." >&2
  exit 2
}
ABC_BIN="$(find_tool "${ABC:-}" yosys-abc abc)" || {
  echo "DEPENDENCIA AUSENTE: ABC. Instale yosys-abc/abc ou defina ABC=/caminho/abc." >&2
  exit 2
}
PYTHON_BIN="$(find_tool "${PYTHON:-}" python3 python)" || {
  echo "DEPENDENCIA AUSENTE: Python 3. Instale-o ou defina PYTHON=/caminho/python." >&2
  exit 2
}

SRC_DIR="$(cd "$(dirname "$SRC")" && pwd)"
SRC_ABS="$SRC_DIR/$(basename "$SRC")"
BASE="${SRC_ABS%.*}"
AIG="$BASE.aig"
SYNTH_LOG="$BASE.yosys.log"
# O parser de comandos do Yosys aceita caminhos POSIX entre aspas.
SRC_YOSYS="${SRC_ABS//\\//}"
AIG_YOSYS="${AIG//\\//}"
SRC_YOSYS="${SRC_YOSYS//\"/\\\"}"
AIG_YOSYS="${AIG_YOSYS//\"/\\\"}"

echo "=== 1/3 sintetizando $SRC -> $AIG"
if ! "$YOSYS_BIN" -p "
  read_verilog -sv \"$SRC_YOSYS\";
  prep -top top -flatten;
  memory_map; async2sync; opt -full;
  techmap; opt -fast;
  abc -fast -g AND;
  setundef -zero; opt_clean;
  write_aiger -zinit \"$AIG_YOSYS\";
  stat
" >"$SYNTH_LOG" 2>&1; then
  echo "ERRO: sintese do Yosys falhou; ultimas linhas de $SYNTH_LOG:" >&2
  tail -n 30 "$SYNTH_LOG" >&2
  exit 2
fi
grep -E '\$_AND_|\$_DFF_P_|ERROR|Warning' "$SYNTH_LOG" || true
if [[ ! -s "$AIG" ]]; then
  echo "ERRO: Yosys terminou sem produzir um AIG nao vazio: $AIG" >&2
  exit 2
fi

echo
echo "=== 2/3 estatisticas do AIG"
if ! "$ABC_BIN" -c "read_aiger \"$AIG_YOSYS\"; print_stats"; then
  echo "ERRO: ABC nao conseguiu ler o AIG." >&2
  exit 2
fi

echo
echo "=== 3/3 IC3/PDR (limite ${TMO}s)"
# /usr/bin/time nao existe em toda imagem; medimos o pico via getrusage
"$PYTHON_BIN" - "$ABC_BIN" "$AIG" "$TMO" <<'PY'
import os
import platform
import re
import subprocess
import sys
import time

try:
    import resource
except ImportError:  # Windows nativo
    resource = None

abc, aig, tmo = sys.argv[1], sys.argv[2], int(sys.argv[3])
t0 = time.time()
cmd = [abc, "-c", f'read_aiger "{aig.replace(os.sep, "/")}"; pdr -v -T {tmo}']
try:
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                       timeout=tmo + 30)
    out = p.stdout + p.stderr
    returncode = p.returncode
except subprocess.TimeoutExpired as exc:
    stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    out = stdout + stderr + "\nWRAPPER TIMEOUT: ABC excedeu T+30 segundos.\n"
    returncode = None
el = time.time() - t0
rss = None
if resource is not None:
    raw_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # Linux informa KiB; macOS informa bytes.
    rss = raw_rss / (1024.0 * 1024.0) if platform.system() == "Darwin" else raw_rss / 1024.0
# a saida BRUTA e sempre preservada: o filtro abaixo ja engoliu um veredito
# inteiro uma vez, e "nenhuma linha casou" e indistinguivel de "nao rodou".
raw = os.path.splitext(aig)[0] + ".abc.out"
with open(raw, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(out)
hits = [ln.strip() for ln in out.splitlines()
        if any(k in ln for k in ("Invariant", "Property proved", "asserted",
                                 "Timeout", "Unfinished", "Verification of",
                                 "Reached", "No output asserted"))]
for ln in hits[-6:]:
    print("   ", ln)
if not hits:
    print("    (ABC nao emitiu linha de veredito — ver", raw, ")")
    print("   ", "\n    ".join(out.strip().splitlines()[-4:]))

lower = out.lower()
verified_invariant = re.search(r"verification of invariant[^\n]*successful", lower)
counterexample = any(
    re.search(r"\boutput\b.*\basserted\b", line) and "no output" not in line
    for line in lower.splitlines()
)
if "property proved" in lower or verified_invariant:
    verdict, exit_code = "PROVADO", 0
elif counterexample:
    verdict, exit_code = "CONTRAEXEMPLO", 1
elif returncode not in (0, None):
    verdict, exit_code = f"ERRO_ABC(codigo={returncode})", 2
else:
    verdict, exit_code = "INCONCLUSIVO", 3

rss_text = f"{rss:.0f}MB" if rss is not None else "indisponivel"
print(f"\n    status={verdict}  tempo={el:.1f}s  pico_RSS={rss_text}  bruto={raw}")
sys.exit(exit_code)
PY
