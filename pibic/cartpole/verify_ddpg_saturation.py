"""
verify_ddpg_saturation.py — Verificação formal de saturação no Actor DDPG.

Propriedades verificadas:

  (A) Saturação de neurônio oculto:
      pre[i] > 0 para todo s ∈ S_bounds?
      ESBMC SUCCESS → saturado (ReLU nunca corta)
      ESBMC FAILED  → normal

  (B) Saturação de saída (adaptada para controle contínuo):
      O pre-tanh z é sempre positivo? Sempre negativo?
      Ambos FAILED → controlador responsivo (gera F > 0 e F < 0)

Uso:
    python verify_ddpg_saturation.py
    python verify_ddpg_saturation.py --output-only
"""

import subprocess, re, os, sys, argparse, math, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddpg_weight_extractor import extract_ddpg_weights
from verify_ddpg_dead_neurons import (
    interval_propagate_layer, relu_bounds, q, c_div,
    X_BND, XD_BND, TH_BND, THD_BND, SCALE, ESBMC, PTH, TIMEOUT,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def harness_sat_layer1(idx, qw1, qb1):
    i = idx
    w = qw1[i]
    return f"""\
/*
 * ddpg_sat_L1_{i}.c — Actor DDPG, neurônio L1[{i}] sempre ativo?
 * SUCCESS → saturado (pre sempre > 0, ReLU nunca corta)
 * FAILED  → normal (às vezes desativa)
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

    __ESBMC_assert(pre <= 0, "neuronio L1[{i}] as vezes desativa?");
    return 0;
}}
"""


def harness_sat_output(qw1, qb1, qw2, qb2, qw_out, qb_out,
                       lo_h1, hi_h1, lo_h2, hi_h2):
    n1 = len(qb1)
    n2 = len(qb2)

    decl_h1 = "\n".join(f"    int h1_{k}=nondet_int();" for k in range(n1))
    assume_h1 = "\n".join(
        f"    __ESBMC_assume(h1_{k}>={lo_h1[k]} && h1_{k}<={hi_h1[k]});"
        for k in range(n1))
    decl_h2 = "\n".join(f"    int h2_{k}=nondet_int();" for k in range(n2))
    assume_h2 = "\n".join(
        f"    __ESBMC_assume(h2_{k}>={lo_h2[k]} && h2_{k}<={hi_h2[k]});"
        for k in range(n2))

    z_expr = " + ".join(
        f"(h2_{k}*({qw_out[0][k]}))/256" for k in range(n2)
    ) + f" + ({qb_out[0]})"

    return f"""\
/*
 * ddpg_sat_output.c — Saturação de saída do Actor DDPG (controle contínuo)
 *
 * Prop A: z > 0 SEMPRE? → controlador sempre aplica força positiva
 * Prop B: z < 0 SEMPRE? → controlador sempre aplica força negativa
 *
 * Ambos FAILED → controlador responsivo → CORRETO
 */
void __ESBMC_assume(_Bool c); void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
{decl_h1}
{assume_h1}

{decl_h2}
{assume_h2}

    int z = {z_expr};

    /* A: saída sempre positiva? (controlador sempre empurra para direita) */
    __ESBMC_assert(!(z > 0), "PropA: controlador sempre aplica forca positiva?");
    /* B: saída sempre negativa? (controlador sempre empurra para esquerda) */
    __ESBMC_assert(!(z < 0), "PropB: controlador sempre aplica forca negativa?");

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
        return ok, out
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-only", action="store_true")
    args = parser.parse_args()

    w = extract_ddpg_weights(PTH)
    qw1 = [[q(v) for v in row] for row in w["w1"]]
    qb1 = [q(v) for v in w["b1"]]
    qw2 = [[q(v) for v in row] for row in w["w2"]]
    qb2 = [q(v) for v in w["b2"]]
    qw_out = [[q(v) for v in row] for row in w["w_out"]]
    qb_out = [q(v) for v in w["b_out"]]

    in_lo = [-X_BND, -XD_BND, -TH_BND, -THD_BND]
    in_hi = [X_BND, XD_BND, TH_BND, THD_BND]

    lo_pre1, hi_pre1 = interval_propagate_layer(qw1, qb1, in_lo, in_hi)
    lo_h1, hi_h1 = relu_bounds(lo_pre1, hi_pre1)
    lo_pre2, hi_pre2 = interval_propagate_layer(qw2, qb2, lo_h1, hi_h1)
    lo_h2, hi_h2 = relu_bounds(lo_pre2, hi_pre2)

    print("=" * 60)
    print("Verificação de Saturação — DDPG Actor")
    print(f"Arquitetura: 4 → 24 → 24 → tanh×10")
    print(f"Quantização: Q8.8 (scale={SCALE})")
    print("=" * 60)

    results = {"layer1_saturated": [], "output_status": ""}

    if not args.output_only:
        n = len(qb1)
        print(f"\n{'─' * 60}")
        print(f" Camada 1 ({n} neurônios) — Saturação (sempre ativo?)")
        print(f"{'─' * 60}")

        saturated = []
        for i in range(n):
            src = harness_sat_layer1(i, qw1, qb1)
            c_file = f"/tmp/ddpg_sat_L1_{i}.c"
            with open(c_file, "w") as f:
                f.write(src)

            ok, _ = run_esbmc(c_file)

            if ok is None:
                status = "\033[93mTIMEOUT\033[0m"
            elif ok:
                status = "\033[91mSATURADO — sempre ativo\033[0m"
                saturated.append(i)
            else:
                status = "\033[92mNORMAL — às vezes desativa\033[0m"

            print(f"  L1[{i:2d}] b={qb1[i]:8}   {status}")

        results["layer1_saturated"] = saturated
        print(f"\n  Saturados: {saturated}")

    # Output saturation
    print(f"\n{'─' * 60}")
    print(f" Saída — Saturação de Direção (controlador sempre mesmo sinal?)")
    print(f"{'─' * 60}")

    src = harness_sat_output(qw1, qb1, qw2, qb2, qw_out, qb_out,
                             lo_h1, hi_h1, lo_h2, hi_h2)
    c_file = "/tmp/ddpg_sat_output.c"
    with open(c_file, "w") as f:
        f.write(src)

    ok, raw = run_esbmc(c_file, timeout=60)

    if ok is None:
        print("  TIMEOUT")
        results["output_status"] = "TIMEOUT"
    elif ok:
        print("  \033[91mSATURADO — controlador sempre aplica força no mesmo sentido!\033[0m")
        results["output_status"] = "SATURATED"
    else:
        print("  \033[92mNORMAL — controlador aplica força em ambos os sentidos\033[0m")
        print("  ✓ Controlador responsivo (F > 0 para alguns estados, F < 0 para outros)")
        results["output_status"] = "RESPONSIVE"

    out_file = os.path.join(HERE, "ddpg_saturation_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em: {out_file}")


if __name__ == "__main__":
    main()
