"""
verify_ddpg_dead_neurons.py — Verificação formal de neurônios mortos no Actor DDPG.

Propriedade verificada:
  Para todo estado válido do Cart-Pole (s ∈ S_bounds),
  o neurônio h[i] = ReLU(pre[i]) é sempre zero?

  ESBMC FAILED   → neurônio VIVO   (existe entrada que o ativa)
  ESBMC SUCCESS  → neurônio MORTO  (nunca ativa — candidato a poda)

Domínio de entrada (Q8.8, scale=256):
  x         ∈ [-2.4, 2.4]   m   → [-614, 614]
  x_dot     ∈ [-5.0, 5.0]   m/s → [-1280, 1280]
  theta     ∈ [-12°, 12°]   rad → [-53,  53]
  theta_dot ∈ [-5.0, 5.0]   rad → [-1280, 1280]

Uso:
    python verify_ddpg_dead_neurons.py          # camada 1
    python verify_ddpg_dead_neurons.py --layer 2
    python verify_ddpg_dead_neurons.py --all
"""

import subprocess, re, os, sys, argparse, math, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddpg_weight_extractor import extract_ddpg_weights

HERE    = os.path.dirname(os.path.abspath(__file__))
ESBMC   = os.path.join(HERE, "../QNNVerifier/esbmc-6.8.0/esbmc")
SCALE   = 256
PTH     = os.path.join(HERE, "ddpg_actor_best.pth")
TIMEOUT = 30

X_BND   = int(2.4    * SCALE)
XD_BND  = int(5.0    * SCALE)
TH_BND  = int(0.2094 * SCALE)
THD_BND = int(5.0    * SCALE)


def q(v):
    return int(round(v * SCALE))


def c_div(a, b):
    return int(a / b)


def interval_propagate_layer(qw, qb, in_lo, in_hi):
    lo_pre, hi_pre = [], []
    for i in range(len(qb)):
        lo, hi = qb[i], qb[i]
        for k in range(len(in_lo)):
            w = qw[i][k]
            if w >= 0:
                lo += c_div(in_lo[k] * w, SCALE)
                hi += c_div(in_hi[k] * w, SCALE)
            else:
                lo += c_div(in_hi[k] * w, SCALE)
                hi += c_div(in_lo[k] * w, SCALE)
        lo_pre.append(lo)
        hi_pre.append(hi)
    return lo_pre, hi_pre


def relu_bounds(lo_pre, hi_pre):
    return [max(0, lo) for lo in lo_pre], [max(0, hi) for hi in hi_pre]


def harness_layer1(idx, qw1, qb1, pre_bound):
    i = idx
    w = qw1[i]
    return f"""\
/*
 * ddpg_dead_L1_{i}.c — Actor DDPG, camada 1, neurônio {i}
 * FAILED  = VIVO  (existe entrada que o ativa)
 * SUCCESS = MORTO (nunca ativa no domínio S_bounds)
 */
void __ESBMC_assume(_Bool c); void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
    int x1 = nondet_int();
    int x2 = nondet_int();
    int x3 = nondet_int();
    int x4 = nondet_int();
    __ESBMC_assume(x1 >= -{X_BND}  && x1 <= {X_BND});
    __ESBMC_assume(x2 >= -{XD_BND} && x2 <= {XD_BND});
    __ESBMC_assume(x3 >= -{TH_BND} && x3 <= {TH_BND});
    __ESBMC_assume(x4 >= -{THD_BND}&& x4 <= {THD_BND});

    int pre = (x1*({w[0]}))/256
            + (x2*({w[1]}))/256
            + (x3*({w[2]}))/256
            + (x4*({w[3]}))/256
            + ({qb1[i]});
    __ESBMC_assume(pre >= -{pre_bound} && pre <= {pre_bound});

    int h = pre > 0 ? pre : 0;
    __ESBMC_assert(h == 0, "neuronio L1[{i}] e sempre morto?");
    return 0;
}}
"""


def harness_layer2(idx, qw2, qb2, lo_h1, hi_h1, pre_bound):
    i = idx
    w = qw2[i]
    decl_h1 = "\n".join(f"    int h1_{k} = nondet_int();" for k in range(len(w)))
    assume_h1 = "\n".join(
        f"    __ESBMC_assume(h1_{k} >= {lo_h1[k]} && h1_{k} <= {hi_h1[k]});"
        for k in range(len(w))
    )
    sum_terms = "\n".join(
        f"    int t{k} = (h1_{k}*({w[k]}))/256;" for k in range(len(w))
    )
    sum_expr = " + ".join(f"t{k}" for k in range(len(w)))

    return f"""\
/*
 * ddpg_dead_L2_{i}.c — Actor DDPG, camada 2, neurônio {i}
 * Usa injeção de intervalo abstrato nos h1.
 * FAILED  = VIVO  | SUCCESS = MORTO
 */
void __ESBMC_assume(_Bool c); void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
{decl_h1}
{assume_h1}

{sum_terms}
    int pre = {sum_expr} + ({qb2[i]});
    __ESBMC_assume(pre >= -{pre_bound} && pre <= {pre_bound});

    int h = pre > 0 ? pre : 0;
    __ESBMC_assert(h == 0, "neuronio L2[{i}] e sempre morto?");
    return 0;
}}
"""


def run_esbmc(c_file, timeout=TIMEOUT):
    try:
        r = subprocess.run(
            [ESBMC, c_file, "--no-unwinding-assertions", "--boolector"],
            capture_output=True, text=True, timeout=timeout,
        )
        out = r.stdout + r.stderr
        ok = "VERIFICATION SUCCESSFUL" in out
        cx = [re.search(rf'\bx{k}\s*=\s*(-?\d+)', out) for k in range(1, 5)]
        ce = ""
        if all(cx):
            vals = [int(m.group(1)) for m in cx]
            ce = "x={:.2f} ẋ={:.2f} θ={:.2f}° θ̇={:.2f}".format(
                vals[0] / SCALE, vals[1] / SCALE,
                vals[2] / SCALE * 180 / math.pi, vals[3] / SCALE)
        return ok, ce
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"


def verify_layer(layer, qw, qb, lo_h=None, hi_h=None, pre_bound=2048):
    n = len(qb)
    print(f"\n{'─' * 58}")
    print(f" Actor DDPG — Camada {layer} ({n} neurônios) — Neurônios Mortos")
    print(f"{'─' * 58}")
    print(f"  {'Neurônio':<10} {'Viés':>8}   {'Resultado'}")
    print(f"  {'─' * 54}")

    dead, alive, timeouts = [], [], []
    for i in range(n):
        if layer == 1:
            src = harness_layer1(i, qw, qb, pre_bound)
        else:
            src = harness_layer2(i, qw, qb, lo_h, hi_h, pre_bound)

        c_file = f"/tmp/ddpg_dead_L{layer}_{i}.c"
        with open(c_file, "w") as f:
            f.write(src)

        ok, ce = run_esbmc(c_file)

        if ok is None:
            status = "\033[93mTIMEOUT\033[0m"
            timeouts.append(i)
        elif ok:
            status = "\033[91mMORTO — VERIFICATION SUCCESSFUL\033[0m"
            dead.append(i)
        else:
            status = "\033[92mVIVO  — VERIFICATION FAILED\033[0m"
            if ce:
                status += f"\n  {'':38}  ({ce})"
            alive.append(i)

        print(f"  L{layer}[{i:2d}]{'':4} b={qb[i]:8}   {status}")

    return dead, alive, timeouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, choices=[1, 2], default=1)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    w = extract_ddpg_weights(PTH)
    qw1 = [[q(v) for v in row] for row in w["w1"]]
    qb1 = [q(v) for v in w["b1"]]
    qw2 = [[q(v) for v in row] for row in w["w2"]]
    qb2 = [q(v) for v in w["b2"]]

    in_lo = [-X_BND, -XD_BND, -TH_BND, -THD_BND]
    in_hi = [X_BND, XD_BND, TH_BND, THD_BND]

    print("=" * 58)
    print("Verificação de Neurônios Mortos — DDPG Actor")
    print(f"Arquitetura: 4 → 24 → 24 → tanh×10")
    print(f"Domínio: x∈[-2.4,2.4] ẋ∈[-5,5] θ∈[-12°,12°] θ̇∈[-5,5]")
    print(f"Quantização: Q8.8 (scale={SCALE})")
    print("=" * 58)

    results = {}
    layers_to_run = [1, 2] if args.all else [args.layer]

    for layer in layers_to_run:
        if layer == 1:
            dead, alive, to = verify_layer(1, qw1, qb1, pre_bound=2048)
        else:
            lo_pre1, hi_pre1 = interval_propagate_layer(qw1, qb1, in_lo, in_hi)
            lo_h1, hi_h1 = relu_bounds(lo_pre1, hi_pre1)
            dead, alive, to = verify_layer(2, qw2, qb2,
                                           lo_h=lo_h1, hi_h=hi_h1,
                                           pre_bound=4096)
        results[f"layer_{layer}"] = {
            "dead": dead, "alive": alive, "timeouts": to,
            "total": len(qb1) if layer == 1 else len(qb2),
        }

    # Save results
    out_file = os.path.join(HERE, "ddpg_dead_neuron_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 58}")
    print("RELATÓRIO FINAL — DDPG Actor")
    print(f"{'=' * 58}")
    for layer_key, r in results.items():
        n = r["total"]
        d = r["dead"]
        print(f"\n{layer_key} ({n} neurônios):")
        if d:
            print(f"  ⚠  MORTOS: {d}")
        else:
            print(f"  ✓  Nenhum neurônio morto")
    print(f"\nResultados salvos em: {out_file}")


if __name__ == "__main__":
    main()
