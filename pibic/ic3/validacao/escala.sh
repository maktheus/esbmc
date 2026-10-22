#!/usr/bin/env bash
# Comparacao controlada: mesma instancia por tamanho, tres motores. Uso: escala.sh "4 8 12 16" TMO
set -u
SIZES="${1:-4 8 12 16}"; TMO="${2:-300}"
mkdir -p escala; cd escala
for H in $SIZES; do
  python3 - "$H" <<'PY'
import json, random, sys
H = int(sys.argv[1]); r = random.Random(7)
rm = lambda n, m, s: [[r.randint(-s, s) for _ in range(m)] for _ in range(n)]
json.dump({"w1": rm(H, 4, 300), "b1": [r.randint(-64, 64) for _ in range(H)],
           "w2": rm(H, H, 200), "b2": [r.randint(-64, 64) for _ in range(H)],
           "w_out": rm(1, H, 200), "b_out": [r.randint(-64, 64)]}, open(f"w{H}.json", "w"))
PY
  python3 ../../gen_transition_system.py --bits 16 --weights w$H.json -o n$H.v >/dev/null
  echo "### H=$H" 
  yosys -q -p "read_verilog -sv n$H.v; prep -top top -flatten; memory_map; async2sync; opt -full; techmap; opt -fast; abc -fast -g AND; setundef -zero; opt_clean; write_aiger -zinit n$H.aig" >/dev/null 2>&1
  yosys-abc -c "read_aiger n$H.aig; print_stats" 2>&1 | grep -E "and =" | sed 's/^/    AIG: /'
  echo "  -- PDR (ABC)";  /usr/bin/time -f "    wall=%es rss=%MKB" yosys-abc -c "read_aiger n$H.aig; pdr -v -T $TMO" 2>&1 | grep -E "Property proved|asserted|Timeout|Unfinished|wall=|Invariant" | tail -3
  echo "  -- BMC (ABC bmc3)"; /usr/bin/time -f "    wall=%es rss=%MKB" yosys-abc -c "read_aiger n$H.aig; bmc3 -F 60 -T $TMO -v" 2>&1 | grep -E "asserted|No output|Reached|Completed|wall=|Timeout|Frame" | tail -3
  echo "  -- Spacer (Z3, nivel de palavra)"
  python3 ../spacer_real.py --weights w$H.json --timeout $TMO 2>&1 | grep -E "resultado|INDECISO" 
done
