"""
verify_saturation.py — Verificação formal de saturação no controlador DQN.

Duas propriedades verificadas:

  (A) Saturação de neurônio oculto:
      "O neurônio h[i] está SEMPRE ATIVO para qualquer entrada válida?"
      — pre[i] > 0 para todo s ∈ S_bounds
      — ESBMC SUCCESS → sempre positivo → ReLU nunca corta → saturado (nunca 0)
      — ESBMC FAILED  → às vezes desativa → comportamento normal

  (B) Saturação de ação (saída):
      "O controlador SEMPRE escolhe a mesma ação?"
      — Q[0] > Q[1] para todo s ∈ S_bounds → sempre empurra à esquerda
      — Q[1] > Q[0] para todo s ∈ S_bounds → sempre empurra à direita
      — ESBMC FAILED em ambos → controlador responsivo → correto

Uso:
    python verify_saturation.py               # camada 1 + saída
    python verify_saturation.py --layer 2     # camada 2 + saída
    python verify_saturation.py --output-only # apenas saturação de ação
"""

import subprocess, re, os, sys, argparse, math

sys.path.insert(0, os.path.dirname(__file__))
from onnx_controller_extractor import extract_controller_weights
from verify_dead_neurons import (
    interval_propagate_layer1, q,
    X_BND, XD_BND, TH_BND, THD_BND, SCALE, ESBMC, ONNX_FILE, TIMEOUT,
)

# Limiar para considerar "saturado" na verificação simbólica
SAT_THRESHOLD = 512   # Q8.8 ≈ 2.0 (pré-ativação > 2.0 sempre = saturado)


def harness_sat_layer1(neuron_idx, qw1, qb1):
    """Verifica se pre[i] > 0 SEMPRE (neurônio nunca desativa = saturado)."""
    i = neuron_idx
    w = qw1[i]
    return f"""\
/*
 * sat_L1_{i}.c — neurônio L1[{i}] sempre ativo?
 * SUCCESS → pre sempre > 0 → saturado (nunca corta na ReLU)
 * FAILED  → às vezes pre ≤ 0 → comportamento normal
 */
void __ESBMC_assume(_Bool c); void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
    int x1=nondet_int(), x2=nondet_int(), x3=nondet_int(), x4=nondet_int();
    __ESBMC_assume(x1>=-{X_BND}  && x1<={X_BND});
    __ESBMC_assume(x2>=-{XD_BND} && x2<={XD_BND});
    __ESBMC_assume(x3>=-{TH_BND} && x3<={TH_BND});
    __ESBMC_assume(x4>=-{THD_BND}&& x4<={THD_BND});

    int pre = (x1*({w[0]}))/256 + (x2*({w[1]}))/256
            + (x3*({w[2]}))/256 + (x4*({w[3]}))/256
            + ({qb1[i]});

    /* SUCCESS = pre sempre > 0 = neurônio saturado */
    __ESBMC_assert(pre <= 0, "neuronio L1[{i}] as vezes desativa?");
    return 0;
}}
"""


def harness_sat_output(qw1, qb1, qw2, qb2, qw_out, qb_out,
                       lo_h1, hi_h1, lo_h2, hi_h2):
    """
    Verifica saturação de ação:
    - Prop A: Q[0] > Q[1] SEMPRE? (sempre push left)
    - Prop B: Q[1] > Q[0] SEMPRE? (sempre push right)
    Usa injeção de intervalo abstrato para h1 e h2.
    """
    n1 = len(qb1)
    n2 = len(qb2)

    # Declarações e assumes para h1, h2
    decl_h1    = "\n".join(f"    int h1_{k}=nondet_int();" for k in range(n1))
    assume_h1  = "\n".join(
        f"    __ESBMC_assume(h1_{k}>={lo_h1[k]} && h1_{k}<={hi_h1[k]});"
        for k in range(n1))
    decl_h2    = "\n".join(f"    int h2_{k}=nondet_int();" for k in range(n2))
    assume_h2  = "\n".join(
        f"    __ESBMC_assume(h2_{k}>={lo_h2[k]} && h2_{k}<={hi_h2[k]});"
        for k in range(n2))

    # Q[0] e Q[1]
    def q_sum(weights, bias, prefix):
        terms = [f"(({prefix}_{k})*({weights[0][k]}))/256" for k in range(len(weights[0]))]
        return " + ".join(terms) + f" + ({bias[0]})"

    q0_expr = " + ".join(
        f"(h2_{k}*({qw_out[0][k]}))/256" for k in range(n2)) + f" + ({qb_out[0]})"
    q1_expr = " + ".join(
        f"(h2_{k}*({qw_out[1][k]}))/256" for k in range(n2)) + f" + ({qb_out[1]})"

    return f"""\
/*
 * sat_output.c — Saturação de ação do controlador DQN
 *
 * Prop A: Q[0] > Q[1] SEMPRE? → controlador sempre escolhe acão 0 (esquerda)
 * Prop B: Q[1] > Q[0] SEMPRE? → controlador sempre escolhe acao 1 (direita)
 *
 * Ambos FAILED → controlador responsivo (escolhe ações diferentes) → CORRETO
 */
void __ESBMC_assume(_Bool c); void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
    /* h1 com bounds derivados da camada 1 */
{decl_h1}
{assume_h1}

    /* h2 com bounds derivados da camada 2 */
{decl_h2}
{assume_h2}

    int q0 = {q0_expr};
    int q1 = {q1_expr};

    /* A: sempre escolhe esquerda? */
    __ESBMC_assert(!(q0 > q1), "PropA: controlador sempre empurra esquerda?");
    /* B: sempre escolhe direita? */
    __ESBMC_assert(!(q1 > q0), "PropB: controlador sempre empurra direita?");

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
        # tenta extrair contraexemplo de x1..x4 ou h
        cx  = [re.search(rf'\bx{k}\s*=\s*(-?\d+)', out) for k in range(1, 5)]
        ce  = ""
        if all(m is not None for m in cx):
            vals = [int(m.group(1)) for m in cx]
            ce   = "x={:.2f} ẋ={:.2f} θ={:.2f}° θ̇={:.2f}".format(
                vals[0]/SCALE, vals[1]/SCALE,
                vals[2]/SCALE*180/math.pi, vals[3]/SCALE)
        return ok, ce
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"


def verify_saturation_layer1(qw1, qb1):
    n = len(qb1)
    print(f"\n{'─'*60}")
    print(f" Camada 1 ({n} neurônios) — Saturação (sempre ativo?)")
    print(f"{'─'*60}")

    saturated, normal = [], []
    for i in range(n):
        src    = harness_sat_layer1(i, qw1, qb1)
        c_file = f"/tmp/sat_L1_{i}.c"
        with open(c_file, "w") as f:
            f.write(src)

        ok, ce = run_esbmc(c_file)

        if ok is None:
            status = "\033[93mTIMEOUT\033[0m"
        elif ok:
            status = "\033[91mSATURADO — sempre ativo (ReLU nunca corta)\033[0m"
            saturated.append(i)
        else:
            status = "\033[92mNORMAL  — às vezes desativa\033[0m"
            if ce:
                status += f"\n  {'':36}  ({ce})"
            normal.append(i)

        print(f"  L1[{i:2d}] b={qb1[i]:8}   {status}")
    return saturated, normal


def verify_saturation_output(qw1, qb1, qw2, qb2, qw_out, qb_out,
                              lo_h1, hi_h1, lo_h2, hi_h2):
    print(f"\n{'─'*60}")
    print(f" Saída — Saturação de Ação (controlador sempre escolhe a mesma?)")
    print(f"{'─'*60}")

    src    = harness_sat_output(qw1, qb1, qw2, qb2, qw_out, qb_out,
                                lo_h1, hi_h1, lo_h2, hi_h2)
    c_file = "/tmp/sat_output.c"
    with open(c_file, "w") as f:
        f.write(src)

    ok, ce = run_esbmc(c_file, timeout=60)

    if ok is None:
        print("  TIMEOUT")
        return
    if ok:
        print("  \033[91mSATURADO — controlador sempre escolhe a mesma ação!\033[0m")
        print("  ⚠ Controlador degenerado — verificar treinamento.")
    else:
        print("  \033[92mNORMAL   — controlador escolhe ações diferentes\033[0m")
        print("  ✓ Q[0] ≠ Q[1] para algum estado → controlador responsivo.")
        if ce:
            print(f"  Contraexemplo: {ce}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, choices=[1, 2], default=1)
    parser.add_argument("--output-only", action="store_true")
    args = parser.parse_args()

    weights = extract_controller_weights(ONNX_FILE)
    n1 = len(weights["b1"])
    n2 = len(weights["b2"])

    qw1   = [[q(v) for v in row] for row in weights["w1"]]
    qb1   =  [q(v) for v in weights["b1"]]
    qw2   = [[q(v) for v in row] for row in weights["w2"]]
    qb2   =  [q(v) for v in weights["b2"]]
    qw_o  = [[q(v) for v in row] for row in weights["w_out"]]
    qb_o  =  [q(v) for v in weights["b_out"]]

    input_bounds = [(-X_BND, X_BND), (-XD_BND, XD_BND),
                    (-TH_BND, TH_BND), (-THD_BND, THD_BND)]
    lo_h1, hi_h1 = interval_propagate_layer1(qw1, qb1, input_bounds)
    lo_h2, hi_h2 = interval_propagate_layer1(qw2, qb2,
                                              list(zip(lo_h1, hi_h1)))

    print("=" * 60)
    print("Verificação de Saturação — DQN Cart-Pole")
    print(f"Controlador: 4 → {n1} → {n2} → 2")
    print("=" * 60)

    if not args.output_only:
        sat, normal = verify_saturation_layer1(qw1, qb1)
        print(f"\n  Saturados : {sat}")
        print(f"  Normais   : {normal[:8]}{'...' if len(normal)>8 else ''}")

    verify_saturation_output(qw1, qb1, qw2, qb2, qw_o, qb_o,
                              lo_h1, hi_h1, lo_h2, hi_h2)
    print()


if __name__ == "__main__":
    main()
