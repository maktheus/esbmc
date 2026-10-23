#!/usr/bin/env python3
"""
parallel_evaluation_28cores.py — Verificacao Formal Paralela em Malha Fechada
utilizando 100% dos processadores logicos (28 threads) nativamente no Windows.

Arquitetura:
1. Deteccao de Hardware (Intel Core i7-14700: 20 nucleos / 28 threads, 64 GB RAM).
2. Bateria 1 — Propriedades Formais do ESBMC em Paralelo (Prop A Right, Prop A Left, Prop B, Prop C, Prop D).
3. Bateria 2 — Domain Slicing (Particionamento do Espaco de Estados em 28 Nucleos Simultaneos):
   Divide o espaco inicial de theta e theta_dot em 28 particoes disjuntas.
   Executa 28 instancias nativas de esbmc.exe concorrentes, uma em cada thread logica do PC.
4. Bateria 3 — Word-Level BMC com Z3 em 28 Processos Concorrentes (Horizontes K=1 a K=10).
5. Consolidacao e Relatorio de Performance (Speedup, CPU utilization, Tempos e Vereditos).
"""

import os
import sys
import json
import time
import math
import tempfile
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configura caminhos
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "pibic"))

from core_verify.esbmc_caller import (
    SAFE, UNSAFE, TIMEOUT, ERROR, run_esbmc as run_esbmc_canonical,
    build_esbmc_cmd, ESBMC_BIN
)
from verify_ddpg_dead_neurons import (
    interval_propagate_layer, relu_bounds, q,
    X_BND, XD_BND, TH_BND, THD_BND, SCALE
)
from verify_ddpg_closed_loop import (
    generate_controller_body, TANH_APPROX_C,
    harness_prop_a_right, harness_prop_a_left, harness_prop_b,
    harness_prop_c, harness_prop_d, DANGER_TH
)

# Carrega pesos Q8.8
WEIGHTS_PATH = os.path.join(HERE, "webapp", "public", "ddpg_weights_q88.json")
with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
    W = json.load(f)

QW1, QB1 = W["w1"], W["b1"]
QW2, QB2 = W["w2"], W["b2"]
QW_OUT, QB_OUT = W["w_out"], W["b_out"]

# Propagacao de intervalos
in_lo = [-X_BND, -XD_BND, -TH_BND, -THD_BND]
in_hi = [X_BND, XD_BND, TH_BND, THD_BND]
lo_pre1, hi_pre1 = interval_propagate_layer(QW1, QB1, in_lo, in_hi)
lo_h1, hi_h1 = relu_bounds(lo_pre1, hi_pre1)
lo_pre2, hi_pre2 = interval_propagate_layer(QW2, QB2, lo_h1, hi_h1)

CTRL_BODY = generate_controller_body(
    QW1, QB1, QW2, QB2, QW_OUT, QB_OUT,
    lo_pre1, hi_pre1, lo_pre2, hi_pre2
)


# ─── Worker para Execucao de Propriedade ESBMC ────────────────────────────────
def run_single_esbmc_task(task_args):
    """Executa uma verificacao ESBMC isolada em um nucleo dedicado."""
    task_id, name, desc, c_code, timeout = task_args
    t_start = time.time()
    
    # Cria arquivo temporario unico para esta thread
    tmp_c = os.path.join(tempfile.gettempdir(), f"task_{task_id}_{name}.c")
    with open(tmp_c, "w", encoding="utf-8") as f:
        f.write(c_code)
        
    res = run_esbmc_canonical(
        tmp_c,
        timeout=timeout,
        no_unwinding_assertions=True,
        z3=True
    )
    duration = time.time() - t_start
    
    # Limpa arquivo temporario
    try:
        os.remove(tmp_c)
    except Exception:
        pass
        
    return {
        "task_id": task_id,
        "name": name,
        "desc": desc,
        "status": res.status,
        "time": duration,
        "rc": res.returncode
    }


# ─── Gerador de Harness com Particionamento (Domain Slicing) ─────────────────
def generate_sliced_harness(slice_idx, total_slices, th_min, th_max):
    """Gera harness da Propriedade B restringindo theta a uma fatia de 1/28."""
    step_size = (th_max - th_min) / total_slices
    s_lo = int(th_min + slice_idx * step_size)
    s_hi = int(th_min + (slice_idx + 1) * step_size)
    
    return f"""\
/*
 * Sliced Harness — Fatia {slice_idx+1}/{total_slices}
 * Dominio restrito: theta in [{s_lo}, {s_hi}]
 */
void __ESBMC_assume(_Bool c);
void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
    int x   = nondet_int();
    int xd  = nondet_int();
    int th  = nondet_int();
    int thd = nondet_int();

    __ESBMC_assume(x   >= -{X_BND}  && x   <= {X_BND});
    __ESBMC_assume(xd  >= -{XD_BND} && xd  <= {XD_BND});
    __ESBMC_assume(th  >= {s_lo}    && th  <= {s_hi});
    __ESBMC_assume(thd >= -{THD_BND}&& thd <= {THD_BND});

{CTRL_BODY}

{TANH_APPROX_C}

    int th_acc  = (4040 * th - 375 * F_Q) / 256;
    int th_new  = th  + (5 * thd) / 256;
    int thd_new = thd + (5 * th_acc) / 256;
    int th_2    = th_new + (5 * thd_new) / 256;

    __ESBMC_assert(th_2 >= -{TH_BND} && th_2 <= {TH_BND},
                   "Fatia {slice_idx}: theta viola regiao segura em 2 passos");
    return 0;
}}
"""


# ─── Worker para Z3 BMC Word-Level ───────────────────────────────────────────
def run_z3_bmc_task(task_args):
    """Executa BMC em nivel de palavra para um horizonte K especifico via Z3."""
    k_horizon, init_bound = task_args
    import z3
    t_start = time.time()
    
    def td_bv(v):
        return z3.If(v >= 0, z3.LShR(v, 8), -z3.LShR(-v, 8))
        
    def relu_bv(v):
        return z3.If(v > 0, v, z3.BitVecVal(0, 32))
        
    def step_bv(x, xd, th, thd):
        h1 = []
        for r, b in zip(QW1, QB1):
            p = td_bv(r[0]*x) + td_bv(r[1]*xd) + td_bv(r[2]*th) + td_bv(r[3]*thd) + b
            h1.append(relu_bv(p))
            
        h2 = []
        for ws, b in zip(QW2, QB2):
            s = z3.BitVecVal(0, 32)
            for w, h in zip(ws, h1):
                s = s + td_bv(w * h)
            h2.append(relu_bv(s + b))
            
        z = z3.BitVecVal(0, 32)
        for w, h in zip(QW_OUT[0], h2):
            z = z + td_bv(w * h)
        z = z + QB_OUT[0]
        
        za = z3.If(z < 0, -z, z)
        ta = z3.If(za <= 64, td_bv(za*252),
             z3.If(za <= 192, 62 + td_bv((za-64)*200),
             z3.If(za <= 384, 162 + td_bv((za-192)*92),
             z3.If(za <= 768, 231 + td_bv((za-384)*16), z3.BitVecVal(255, 32)))))
        tz = z3.If(z < 0, -ta, ta)
        FQ = tz * 10
        tacc = td_bv(4040*th - 375*FQ)
        return x + td_bv(5*xd), xd + td_bv(5*FQ), th + td_bv(5*thd), thd + td_bv(5*tacc)

    s = z3.Solver()
    s.set("timeout", 180000) # 180s
    
    x, xd, th, thd = [z3.BitVec(n, 32) for n in ['x0', 'xd0', 'th0', 'thd0']]
    b = init_bound
    s.add(x >= -b, x <= b, xd >= -b, xd <= b, th >= -b, th <= b, thd >= -b, thd <= b)
    
    for _ in range(k_horizon):
        x, xd, th, thd = step_bv(x, xd, th, thd)
        
    # Condicao de falha: |th| > 53 ou |x| > 614
    s.add(z3.Or(th > 53, th < -53, x > 614, x < -614))
    
    check_res = s.check()
    duration = time.time() - t_start
    
    status_str = "SAFE (UNSAT)" if check_res == z3.unsat else ("UNSAFE (SAT)" if check_res == z3.sat else "TIMEOUT/UNKNOWN")
    return {
        "k": k_horizon,
        "status": status_str,
        "time": duration
    }


# ─── Execucao Principal ──────────────────────────────────────────────────────
def main():
    if hasattr(sys.stdout, "reconfigure"):
        try: sys.stdout.reconfigure(encoding="utf-8")
        except Exception: pass
        
    num_cpus = os.cpu_count() or 28
    mem_total_gb = psutil.virtual_memory().total / (1024**3)
    
    print("=" * 80)
    print(" MOTOR DE VERIFICACAO FORMAL MULTI-CORE NATIVO — ESBMC & SMT")
    print(f" Hardware Detectado: {num_cpus} Nucleos Logicos | {mem_total_gb:.1f} GB RAM")
    print(f" Binario ESBMC: {ESBMC_BIN}")
    print("=" * 80)
    
    # ─── FASE 1: 5 Propriedades Formais Concorrentes (ESBMC) ──────────────────
    print(f"\n[FASE 1] Executando as 5 Propriedades Formais do ESBMC em Paralelo NATIVO...")
    t0_fase1 = time.time()
    
    tasks_fase1 = [
        (1, "PropA_Right", "Propriedade A — Direcao Positiva (th > 25 -> F > 0)", harness_prop_a_right(CTRL_BODY), 90),
        (2, "PropA_Left",  "Propriedade A — Direcao Negativa (th < -25 -> F < 0)", harness_prop_a_left(CTRL_BODY), 90),
        (3, "PropB_2step", "Propriedade B — Seguranca Malha Fechada 2 Passos", harness_prop_b(CTRL_BODY), 90),
        (4, "PropC_Bounds", "Propriedade C — Sanidade Quantizacao (|F| <= 10N)", harness_prop_c(CTRL_BODY), 90),
        (5, "PropD_Danger", "Propriedade D — Forca Minima na Zona de Perigo", harness_prop_d(CTRL_BODY), 90),
    ]
    
    results_fase1 = []
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_single_esbmc_task, task) for task in tasks_fase1]
        for f in as_completed(futures):
            res = f.result()
            results_fase1.append(res)
            print(f"  -> Concluido: {res['name']:<15} | Status: {res['status']:<10} | Tempo: {res['time']:.2f}s")
            
    wall_fase1 = time.time() - t0_fase1
    total_cpu_fase1 = sum(r['time'] for r in results_fase1)
    speedup_fase1 = total_cpu_fase1 / wall_fase1 if wall_fase1 > 0 else 1.0
    print(f"Fase 1 Concluida em {wall_fase1:.2f}s (Tempo CPU Somado: {total_cpu_fase1:.2f}s | Speedup: {speedup_fase1:.2f}x)")

    # ─── FASE 2: Domain Slicing em 28 Threads Concorrentes (100% CPU) ──────────
    print(f"\n[FASE 2] Disparando Domain Slicing em {num_cpus} Nucleos Simultaneos (100% CPU)...")
    print(f"Particionando theta in [-53, 53] em {num_cpus} sub-dominios disjuntos...")
    t0_fase2 = time.time()
    
    tasks_fase2 = []
    for i in range(num_cpus):
        c_code = generate_sliced_harness(i, num_cpus, -TH_BND, TH_BND)
        tasks_fase2.append((i+1, f"Slice_{i+1:02d}", f"Fatia {i+1}/{num_cpus}", c_code, 90))
        
    results_fase2 = []
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [executor.submit(run_single_esbmc_task, task) for task in tasks_fase2]
        for f in as_completed(futures):
            res = f.result()
            results_fase2.append(res)
            print(f"  [Core {res['task_id']:02d}/{num_cpus}] {res['name']} finalizado: {res['status']:<8} em {res['time']:.2f}s")
            
    wall_fase2 = time.time() - t0_fase2
    total_cpu_fase2 = sum(r['time'] for r in results_fase2)
    speedup_fase2 = total_cpu_fase2 / wall_fase2 if wall_fase2 > 0 else 1.0
    print(f"Fase 2 Concluida em {wall_fase2:.2f}s (Tempo CPU Somado: {total_cpu_fase2:.2f}s | Speedup: {speedup_fase2:.2f}x)")
    
    # ─── FASE 3: BMC Word-Level Paralelo (Horizontes K=1 a K=10) ──────────────
    print(f"\n[FASE 3] Executando Bounded Model Checking Word-Level (K=1 a K=10) em Paralelo com Z3...")
    t0_fase3 = time.time()
    
    tasks_fase3 = [(k, 5) for k in range(1, 11)]
    results_fase3 = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_z3_bmc_task, task) for task in tasks_fase3]
        for f in as_completed(futures):
            res = f.result()
            results_fase3.append(res)
            print(f"  [Horizonte K={res['k']:02d}] Status: {res['status']:<15} | Tempo: {res['time']:.2f}s")
            
    wall_fase3 = time.time() - t0_fase3
    total_cpu_fase3 = sum(r['time'] for r in results_fase3)
    speedup_fase3 = total_cpu_fase3 / wall_fase3 if wall_fase3 > 0 else 1.0
    print(f"Fase 3 Concluida em {wall_fase3:.2f}s (Tempo CPU Somado: {total_cpu_fase3:.2f}s | Speedup: {speedup_fase3:.2f}x)")
    
    # ─── RELATORIO FINAL CONSOLIDADO ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" RESUMO DA AVALIACAO MULTI-CORE (100% HARDWARE NATIVO NO WINDOWS)")
    print("=" * 80)
    print(f"Processador: {num_cpus} threads logicas | Memoria: {mem_total_gb:.1f} GB")
    print(f"Total de Tarefas Formais Executadas: {len(results_fase1) + len(results_fase2) + len(results_fase3)}")
    total_wall = wall_fase1 + wall_fase2 + wall_fase3
    total_cpu = total_cpu_fase1 + total_cpu_fase2 + total_cpu_fase3
    print(f"Tempo Real Decorrido (Wall-clock): {total_wall:.2f} segundos")
    print(f"Trabalho Total de CPU Processado:  {total_cpu:.2f} segundos")
    print(f"Fator de Aceleracao Global:        {total_cpu / total_wall:.2f}x mais rapido que 1 unico nucleo!")
    print("=" * 80)
    
    # Salva json com resultados para integracao
    out_json = {
        "hardware": {
            "cpus": num_cpus,
            "memory_gb": mem_total_gb
        },
        "fase1_properties": results_fase1,
        "fase2_domain_slicing": results_fase2,
        "fase3_word_level_bmc": results_fase3,
        "metrics": {
            "wall_time": total_wall,
            "cpu_time": total_cpu,
            "speedup": total_cpu / total_wall
        }
    }
    with open(os.path.join(HERE, "resultado_paralelo_28cores.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"Resultados salvos em: {os.path.join(HERE, 'resultado_paralelo_28cores.json')}")

if __name__ == "__main__":
    main()
