#!/usr/bin/env bash
# run_pdr.sh — Sintetiza o sistema de transição e prova a propriedade de
# segurança com IC3/PDR (ABC), reportando tempo e pico de memória reais.
#
#   ./run_pdr.sh cl_ddpg16.v [segundos]
#
# Requer: yosys (traz o yosys-abc embutido).  apt-get install -y yosys
#
# Saída do ABC:
#   "Property proved"      -> invariante indutivo encontrado, seguro para SEMPRE
#   "Output ... asserted"  -> contraexemplo concreto no passo indicado
#   "Timeout"/"Unfinished" -> indeciso; nao confundir com "seguro"
set -euo pipefail

SRC="${1:?uso: run_pdr.sh <arquivo.v> [timeout_s]}"
TMO="${2:-1800}"
BASE="${SRC%.v}"
AIG="$BASE.aig"

echo "=== 1/3 sintetizando $SRC -> $AIG"
yosys -p "
  read_verilog -sv $SRC;
  prep -top top -flatten;
  memory_map; async2sync; opt -full;
  techmap; opt -fast;
  abc -fast -g AND;
  setundef -zero; opt_clean;
  write_aiger -zinit $AIG;
  stat
" 2>&1 | grep -E '\$_AND_|\$_DFF_P_|ERROR|Warning' || true

echo
echo "=== 2/3 estatisticas do AIG"
yosys-abc -c "read_aiger $AIG; print_stats"

echo
echo "=== 3/3 IC3/PDR (limite ${TMO}s)"
# /usr/bin/time nao existe em toda imagem; medimos o pico via getrusage
python3 - "$AIG" "$TMO" <<'PY'
import resource, subprocess, sys, time
aig, tmo = sys.argv[1], sys.argv[2]
t0 = time.time()
p = subprocess.run(["yosys-abc", "-c", f"read_aiger {aig}; pdr -v -T {tmo}"],
                   capture_output=True, text=True, errors="replace")
el = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
out = p.stdout + p.stderr
for ln in out.splitlines():
    if any(k in ln for k in ("Invariant", "Property proved", "asserted",
                             "Timeout", "Unfinished", "Verification of")):
        print("   ", ln.strip())
print(f"\n    tempo={el:.1f}s  pico_RSS={rss:.0f}MB")
PY
