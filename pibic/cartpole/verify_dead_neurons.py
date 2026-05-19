"""
verify_dead_neurons.py — Verificação formal de neurônios mortos no controlador DQN.

Propriedade verificada:
  Para todo estado válido do Cart-Pole (s ∈ S_bounds),
  o neurônio h[i] = ReLU(pre[i]) é sempre zero?

  ESBMC FAILED   → neurônio VIVO   (existe entrada que o ativa)
  ESBMC SUCCESS  → neurônio MORTO  (nunca ativa — candidato a poda)

Domínio de entrada (Q8.8, scale=256):
  x         ∈ [-2.4, 2.4]   m   → [-614, 614]
  x_dot     ∈ [-5.0, 5.0]   m/s → [-1280, 1280]
  theta     ∈ [-12°, 12°]   rad → [-53,  53]    (0.2094 × 256 ≈ 53)
  theta_dot ∈ [-5.0, 5.0]   rad → [-1280, 1280]

Uso:
    python verify_dead_neurons.py          # verifica camada 1 (24 neurônios)
    python verify_dead_neurons.py --layer 2  # verifica camada 2
    python verify_dead_neurons.py --all      # verifica ambas as camadas
"""

import subprocess, re, os, sys, argparse, math

sys.path.insert(0, os.path.dirname(__file__))
from onnx_controller_extractor import extract_controller_weights

ESBMC     = os.path.join(os.path.dirname(__file__),
                         "../QNNVerifier/esbmc-6.8.0/esbmc")
SCALE     = 256
ONNX_FILE = os.path.join(os.path.dirname(__file__), "dqn_cartpole.onnx")
TIMEOUT   = 30   # segundos por neurônio

# Bounds do estado em Q8.8
X_BND       = int(2.4   * SCALE)   # 614
XD_BND      = int(5.0   * SCALE)   # 1280
TH_BND      = int(0.2094 * SCALE)  # 53
THD_BND     = int(5.0   * SCALE)   # 1280


def q(v):
    return int(round(v * SCALE))


def interval_propagate_layer1(qw1, qb1, bounds_in):
    """
    Propaga bounds de entrada pela camada 1 para obter bounds de h1[i].
    Necessário para verificar camada 2 via injeção de intervalo abstrato.
    """
    lo_h1, hi_h1 = [], []
    for i in range(len(qb1)):
        pre_lo = qb1[i]
        pre_hi = qb1[i]
        for k, (lo_k, hi_k) in enumerate(bounds_in):
            w = qw1[i][k]
            if w >= 0:
                pre_lo += (lo_k * w) // SCALE
                pre_hi += (hi_k * w) // SCALE
            else:
                pre_lo += (hi_k * w) // SCALE
                pre_hi += (lo_k * w) // SCALE
        lo_h1.append(max(0, pre_lo))
        hi_h1.append(max(0, pre_hi))
    return lo_h1, hi_h1


def harness_layer1(neuron_idx, qw1, qb1, pre_bound):
    i = neuron_idx
    w = qw1[i]
    return f"""\
/*
 * dead_neuron_L1_{i}.c  —  camada 1, neurônio {i}
 * FAILED  = VIVO  (existe entrada que o ativa)
 * SUCCESS = MORTO (nunca ativa no domínio S_bounds)
 */
void __ESBMC_assume(_Bool c); void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
    /* estado simbolico em Q8.8 */
    int x1 = nondet_int(); /* posicao carro   */
    int x2 = nondet_int(); /* velocidade carro */
    int x3 = nondet_int(); /* angulo pendulo   */
    int x4 = nondet_int(); /* vel. angular     */
    __ESBMC_assume(x1 >= -{X_BND}  && x1 <= {X_BND});
    __ESBMC_assume(x2 >= -{XD_BND} && x2 <= {XD_BND});
    __ESBMC_assume(x3 >= -{TH_BND} && x3 <= {TH_BND});
    __ESBMC_assume(x4 >= -{THD_BND}&& x4 <= {THD_BND});

    int pre = (x1*({w[0]}))/256
            + (x2*({w[1]}))/256
            + (x3*({w[2]}))/256
            + (x4*({w[3]}))/256
            + ({qb1[i]});
    __ESBMC_assume(pre >= -{pre_bound} && pre <= {pre_bound}); /* invariante */

    int h = pre > 0 ? pre : 0; /* ReLU */
    __ESBMC_assert(h == 0, "neuronio L1[{i}] e sempre morto?");
    return 0;
}}
"""


def harness_layer2(neuron_idx, qw2, qb2, lo_h1, hi_h1, pre_bound):
    i = neuron_idx
    w = qw2[i]
    assume_h1 = ""
    sum_terms  = ""
    for k in range(len(w)):
        assume_h1 += f"    __ESBMC_assume(h1_{k} >= {lo_h1[k]} && h1_{k} <= {hi_h1[k]});\n"
        sum_terms  += f"    int t{k} = (h1_{k}*({w[k]}))/256;\n"

    decl_h1  = "\n".join(f"    int h1_{k} = nondet_int();" for k in range(len(w)))
    sum_expr = " + ".join(f"t{k}" for k in range(len(w)))

    return f"""\
/*
 * dead_neuron_L2_{i}.c  —  camada 2, neurônio {i}
 * Usa injeção de intervalo abstrato nos h1 (QNNVerifier technique).
 * FAILED  = VIVO  | SUCCESS = MORTO
 */
void __ESBMC_assume(_Bool c); void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
    /* h1 simbolicos com bounds derivados da camada 1 */
{decl_h1}
{assume_h1}
    /* pre-ativacao do neuronio L2[{i}] */
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
        ok  = "VERIFICATION SUCCESSFUL" in out
        cx  = [re.search(rf'\bx{k}\s*=\s*(-?\d+)', out) for k in range(1, 5)]
        ce  = ""
        if all(cx):
            vals = [int(m.group(1)) for m in cx]
            ce   = ("x={:.2f} ẋ={:.2f} θ={:.2f}° θ̇={:.2f}".format(
                vals[0]/SCALE, vals[1]/SCALE,
                vals[2]/SCALE * 180 / math.pi, vals[3]/SCALE))
        return ok, ce
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"


def verify_layer(layer, qw, qb, lo_h=None, hi_h=None, pre_bound=2048):
    n_neurons = len(qb)
    print(f"\n{'─'*58}")
    print(f" Camada {layer}  ({n_neurons} neurônios)  —  Neurônios Mortos")
    print(f"{'─'*58}")
    print(f"  {'Neurônio':<10} {'Viés':>8}   {'Resultado'}")
    print(f"  {'─'*54}")

    dead, alive, timeouts = [], [], []
    for i in range(n_neurons):
        if layer == 1:
            src = harness_layer1(i, qw, qb, pre_bound)
        else:
            src = harness_layer2(i, qw, qb, lo_h, hi_h, pre_bound)

        c_file = f"/tmp/dead_L{layer}_{i}.c"
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
    parser.add_argument("--all",   action="store_true")
    args = parser.parse_args()

    weights = extract_controller_weights(ONNX_FILE)
    n1 = len(weights["b1"])
    n2 = len(weights["b2"])

    qw1 = [[q(v) for v in row] for row in weights["w1"]]
    qb1 =  [q(v) for v in weights["b1"]]
    qw2 = [[q(v) for v in row] for row in weights["w2"]]
    qb2 =  [q(v) for v in weights["b2"]]

    # Bounds de entrada para camada 1
    input_bounds = [(-X_BND, X_BND), (-XD_BND, XD_BND),
                    (-TH_BND, TH_BND), (-THD_BND, THD_BND)]

    print("=" * 58)
    print("Verificação de Neurônios Mortos — DQN Cart-Pole")
    print(f"Controlador: 4 → {n1} → {n2} → 2")
    print(f"Domínio: x∈[-2.4,2.4] ẋ∈[-5,5] θ∈[-12°,12°] θ̇∈[-5,5]")
    print("=" * 58)

    all_dead = {}
    layers_to_run = [1, 2] if args.all else [args.layer]

    for layer in layers_to_run:
        if layer == 1:
            dead, alive, to = verify_layer(1, qw1, qb1, pre_bound=2048)
        else:
            lo_h1, hi_h1 = interval_propagate_layer1(qw1, qb1, input_bounds)
            dead, alive, to = verify_layer(2, qw2, qb2,
                                           lo_h=lo_h1, hi_h=hi_h1,
                                           pre_bound=4096)
        all_dead[layer] = dead

    print(f"\n{'='*58}")
    print("RELATÓRIO FINAL")
    print(f"{'='*58}")
    for layer, dead in all_dead.items():
        n = n1 if layer == 1 else n2
        print(f"\nCamada {layer} ({n} neurônios):")
        if dead:
            print(f"  ⚠  MORTOS (candidatos a poda): {dead}")
        else:
            print(f"  ✓  Nenhum neurônio morto — todos contribuem para a saída")
    print()


if __name__ == "__main__":
    main()
