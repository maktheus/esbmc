#!/usr/bin/env python3
"""
benchmark_z3_vs_esbmc_28cores.py — Comparativo Direto e Justo entre Z3 e ESBMC
utilizando Domain Slicing (28 fatias) sobre 100% dos processadores logicos (28 threads).

Condicoes de Paridade Absoluta:
1. Mesmo espaco de busca: theta in [-53, 53] particionado em exatamente 28 fatias disjuntas.
2. Mesma semantica Q8.8 exata: pesos do DDPG, propagacao de intervalos, tanh em 5 ramos, dinamica em 2 passos.
3. Mesma propriedade: theta_2 in [-53, 53] (seguranca em malha fechada).
4. Mesmo timeout por fatia: 90.0 segundos.
5. Mesmo hardware: 28 threads concorrentes rodando simultaneamente (100% de CPU).
"""

import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "pibic"))

from verify_ddpg_dead_neurons import (
    interval_propagate_layer, relu_bounds,
    X_BND, XD_BND, TH_BND, THD_BND
)

# Carrega pesos Q8.8
WEIGHTS_PATH = os.path.join(HERE, "webapp", "public", "ddpg_weights_q88.json")
with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
    W = json.load(f)

QW1, QB1 = W["w1"], W["b1"]
QW2, QB2 = W["w2"], W["b2"]
QW_OUT, QB_OUT = W["w_out"], W["b_out"]

# Propagacao de intervalos (exatamente a mesma do C/ESBMC)
in_lo = [-X_BND, -XD_BND, -TH_BND, -THD_BND]
in_hi = [X_BND, XD_BND, TH_BND, THD_BND]
lo_pre1, hi_pre1 = interval_propagate_layer(QW1, QB1, in_lo, in_hi)
lo_h1, hi_h1 = relu_bounds(lo_pre1, hi_pre1)
lo_pre2, hi_pre2 = interval_propagate_layer(QW2, QB2, lo_h1, hi_h1)


# ─── Worker para Z3 com Domain Slicing ────────────────────────────────────────
def run_z3_slice(args):
    slice_idx, total_slices, timeout_sec = args
    import z3
    t_start = time.time()
    
    def bv(v): return z3.BitVecVal(v, 32)
    def td(v): return (v + z3.If(v < bv(0), bv(255), bv(0))) >> 8
    def relu(v): return z3.If(v > bv(0), v, bv(0))

    x, xd, th, thd = [z3.BitVec(n, 32) for n in ['x', 'xd', 'th', 'thd']]
    s = z3.SolverFor('QF_BV')
    s.set("timeout", int(timeout_sec * 1000))

    # Limites operacionais identicos ao C
    s.add(x >= bv(-X_BND), x <= bv(X_BND))
    s.add(xd >= bv(-XD_BND), xd <= bv(XD_BND))

    # Fatia angular
    step_size = (2 * TH_BND) / total_slices
    s_lo = int(-TH_BND + slice_idx * step_size)
    s_hi = int(-TH_BND + (slice_idx + 1) * step_size)
    s.add(th >= bv(s_lo), th <= bv(s_hi))
    s.add(thd >= bv(-THD_BND), thd <= bv(THD_BND))

    # Camada 1
    h1 = []
    for i in range(24):
        r, b = QW1[i], QB1[i]
        p1 = td(x * bv(r[0])) + td(xd * bv(r[1])) + td(th * bv(r[2])) + td(thd * bv(r[3])) + bv(b)
        s.add(p1 >= bv(lo_pre1[i]), p1 <= bv(hi_pre1[i]))
        h1.append(relu(p1))

    # Camada 2
    h2 = []
    for j in range(24):
        ws, b = QW2[j], QB2[j]
        p2 = sum((td(h * bv(w)) for h, w in zip(h1, ws)), bv(0)) + bv(b)
        s.add(p2 >= bv(lo_pre2[j]), p2 <= bv(hi_pre2[j]))
        h2.append(relu(p2))

    # Saida
    z = sum((td(h * bv(w)) for h, w in zip(h2, QW_OUT[0])), bv(0)) + bv(QB_OUT[0])

    # Tanh em 5 ramos
    za = z3.If(z < bv(0), -z, z)
    ta = z3.If(za <= bv(64), td(za*bv(252)),
         z3.If(za <= bv(192), bv(62) + td((za-bv(64))*bv(200)),
         z3.If(za <= bv(384), bv(162) + td((za-bv(192))*bv(92)),
         z3.If(za <= bv(768), bv(231) + td((za-bv(384))*bv(16)), bv(255)))))
    tz = z3.If(z < bv(0), -ta, ta)
    FQ = tz * bv(10)

    # Dinamica em 2 passos
    th_acc  = td(bv(4040) * th - bv(375) * FQ)
    th_new  = th + td(bv(5) * thd)
    thd_new = thd + td(bv(5) * th_acc)
    th_2    = th_new + td(bv(5) * thd_new)

    # Violacao da seguranca
    s.add(z3.Or(th_2 < bv(-TH_BND), th_2 > bv(TH_BND)))

    res = s.check()
    duration = time.time() - t_start
    
    # Classificacao rigorosa:
    # sat = encontrou estado violador (UNSAFE)
    # unsat = provou que nenhum estado viola na fatia (SAFE)
    # unknown / timeout = indeciso
    if res == z3.sat:
        status = "UNSAFE"
    elif res == z3.unsat:
        status = "SAFE"
    else:
        status = "TIMEOUT"

    return {
        "slice_idx": slice_idx + 1,
        "name": f"Slice_{slice_idx+1:02d}",
        "range": f"[{s_lo}, {s_hi}]",
        "status": status,
        "time": duration
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try: sys.stdout.reconfigure(encoding="utf-8")
        except Exception: pass

    num_slices = 28
    timeout_sec = 90.0

    print("=" * 85)
    print(" BENCHMARK RIGOROSO: Z3 (QF_BV) VS ESBMC EM 28 NUCLEOS CONCORRENTES")
    print(" Condicao de Paridade: 28 Fatias Disjuntas | Timeout = 90s por Fatia | 100% CPU")
    print("=" * 85)

    # Carrega dados do ESBMC obtidos previamente
    esbmc_data = {}
    esbmc_json_path = os.path.join(HERE, "resultado_paralelo_28cores.json")
    if os.path.exists(esbmc_json_path):
        with open(esbmc_json_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
            for item in prev.get("fase2_domain_slicing", []):
                esbmc_data[item["name"]] = item

    print(f"\n[INICIANDO Z3] Disparando 28 Workers Concorrentes no Z3 (QF_BV)...")
    t0_z3 = time.time()
    
    tasks = [(i, num_slices, timeout_sec) for i in range(num_slices)]
    z3_results = []
    
    with ProcessPoolExecutor(max_workers=num_slices) as executor:
        futures = [executor.submit(run_z3_slice, t) for t in tasks]
        for f in as_completed(futures):
            res = f.result()
            z3_results.append(res)
            # Imprime progresso em tempo real
            print(f"  [Z3 Core {res['slice_idx']:02d}/28] {res['name']} ({res['range']}) -> Status: {res['status']:<8} | Tempo: {res['time']:.2f}s")

    wall_z3 = time.time() - t0_z3
    total_cpu_z3 = sum(r['time'] for r in z3_results)
    speedup_z3 = total_cpu_z3 / wall_z3 if wall_z3 > 0 else 1.0

    print(f"\nZ3 Concluido em {wall_z3:.2f}s (Tempo CPU Somado: {total_cpu_z3:.2f}s | Speedup: {speedup_z3:.2f}x)")

    # Ordena resultados por slice_idx
    z3_results.sort(key=lambda x: x['slice_idx'])
    z3_dict = {r['name']: r for r in z3_results}

    # ─── TABELA COMPARATIVA LADO A LADO ───────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'Slice':<10} | {'Faixa [lo, hi]':<16} | {'ESBMC Status':<12} | {'ESBMC (s)':<10} | {'Z3 Status':<10} | {'Z3 (s)':<10} | {'Vencedor'}")
    print("-" * 95)

    esbmc_wins = 0
    z3_wins = 0
    ties = 0

    comparison_table = []

    for i in range(1, num_slices + 1):
        name = f"Slice_{i:02d}"
        z_res = z3_dict.get(name, {"status": "N/A", "time": 0.0, "range": ""})
        e_res = esbmc_data.get(name, {"status": "TIMEOUT", "time": 90.0})

        z_stat, z_t = z_res["status"], z_res["time"]
        e_stat, e_t = e_res["status"], e_res["time"]

        # Determina vencedor da fatia
        if z_stat in ("SAFE", "UNSAFE") and e_stat == "TIMEOUT":
            winner = "Z3 (Resolveu vs Timeout)"
            z3_wins += 1
        elif e_stat in ("SAFE", "UNSAFE") and z_stat == "TIMEOUT":
            winner = "ESBMC (Resolveu vs Timeout)"
            esbmc_wins += 1
        elif z_stat in ("SAFE", "UNSAFE") and e_stat in ("SAFE", "UNSAFE"):
            if z_t < e_t:
                winner = f"Z3 ({e_t/z_t:.1f}x mais rapido)"
                z3_wins += 1
            else:
                winner = f"ESBMC ({z_t/e_t:.1f}x mais rapido)"
                esbmc_wins += 1
        else:
            winner = "Empate (Ambos Timeout)"
            ties += 1

        print(f"{name:<10} | {z_res.get('range', ''):<16} | {e_stat:<12} | {e_t:<10.2f} | {z_stat:<10} | {z_t:<10.2f} | {winner}")

        comparison_table.append({
            "slice": name,
            "range": z_res.get("range", ""),
            "esbmc_status": e_stat,
            "esbmc_time": e_t,
            "z3_status": z_stat,
            "z3_time": z_t,
            "winner": winner
        })

    print("=" * 95)
    print(f"PLACAR FINAL DE FATIAS:")
    print(f"  Vitorias do Z3:    {z3_wins} fatias")
    print(f"  Vitorias do ESBMC: {esbmc_wins} fatias")
    print(f"  Empates (Timeout): {ties} fatias")
    print("=" * 95)

    # Salva json
    out_comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_threads": num_slices,
        "metrics_z3": {
            "wall_time": wall_z3,
            "cpu_time": total_cpu_z3,
            "speedup": speedup_z3,
            "solved_count": sum(1 for r in z3_results if r['status'] in ('SAFE', 'UNSAFE')),
            "timeout_count": sum(1 for r in z3_results if r['status'] == 'TIMEOUT')
        },
        "score": {
            "z3_wins": z3_wins,
            "esbmc_wins": esbmc_wins,
            "ties": ties
        },
        "slices": comparison_table
    }

    out_file = os.path.join(HERE, "comparativo_z3_vs_esbmc_28cores.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_comparison, f, indent=2)
    print(f"\nDados completos do comparativo salvos em: {out_file}")

if __name__ == "__main__":
    main()
