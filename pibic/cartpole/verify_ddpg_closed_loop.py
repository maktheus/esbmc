"""
verify_ddpg_closed_loop.py — Verificação formal em malha fechada do Actor DDPG.

Propriedades verificadas:

  Property A — Direção: θ > threshold ∧ θ̇ ≥ 0 → F > 0 (via monotonicidade de tanh)
    Usa z > 0 ⟺ tanh(z) > 0 ⟺ F > 0. Sem necessidade de aproximar tanh.

  Property B — Segurança 1-step: s₀ ∈ S_safe → |θ_new| ≤ 12° após 1 passo.
    Usa aproximação linear de tanh (mesma do browser) + dinâmica linearizada.

  Property C — Bounds de saída: |F| ≤ 10 N sempre (sanidade Q8.8 + tanh).

O controlador no browser usa a MESMA aritmética Q8.8 e a MESMA
aproximação de tanh → contraexemplos reproduzem exatamente.

Uso:
    python verify_ddpg_closed_loop.py
"""

import subprocess, re, os, sys, json, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddpg_weight_extractor import extract_ddpg_weights
from verify_ddpg_dead_neurons import (
    interval_propagate_layer, relu_bounds, q, c_div,
    X_BND, XD_BND, TH_BND, THD_BND, SCALE, ESBMC, PTH,
)

HERE    = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 120
DANGER_TH = int(0.10 * SCALE)  # 25 ≈ 5.7°


def generate_controller_body(qw1, qb1, qw2, qb2, qw_out, qb_out,
                             lo_pre1, hi_pre1, lo_pre2, hi_pre2):
    lines = []

    for i in range(24):
        w0, w1, w2, w3 = qw1[i]
        b = qb1[i]
        lines.append(
            f"    int pre1_{i} = (x*({w0}))/256 + (xd*({w1}))/256"
            f" + (th*({w2}))/256 + (thd*({w3}))/256 + ({b});"
        )
        lines.append(
            f"    __ESBMC_assume(pre1_{i} >= {lo_pre1[i]} && pre1_{i} <= {hi_pre1[i]});"
        )
        lines.append(f"    int h1_{i} = pre1_{i} > 0 ? pre1_{i} : 0;")

    lines.append("")

    for j in range(24):
        terms = " + ".join(f"(h1_{k}*({qw2[j][k]}))/256" for k in range(24))
        b = qb2[j]
        lines.append(f"    int pre2_{j} = {terms} + ({b});")
        lines.append(
            f"    __ESBMC_assume(pre2_{j} >= {lo_pre2[j]} && pre2_{j} <= {hi_pre2[j]});"
        )
        lines.append(f"    int h2_{j} = pre2_{j} > 0 ? pre2_{j} : 0;")

    lines.append("")

    out_terms = " + ".join(f"(h2_{k}*({qw_out[0][k]}))/256" for k in range(24))
    lines.append(f"    int z = {out_terms} + ({qb_out[0]});")

    return "\n".join(lines)


TANH_APPROX_C = """\
    /* Aproximação linear de tanh em Q8.8 — mesma do browser TypeScript */
    int z_abs = z >= 0 ? z : -z;
    int tanh_abs;
    if (z_abs <= 64)        tanh_abs = (z_abs * 252) / 256;
    else if (z_abs <= 192)  tanh_abs = 62 + ((z_abs - 64) * 200) / 256;
    else if (z_abs <= 384)  tanh_abs = 162 + ((z_abs - 192) * 92) / 256;
    else if (z_abs <= 768)  tanh_abs = 231 + ((z_abs - 384) * 16) / 256;
    else                    tanh_abs = 255;
    int tanh_z = z >= 0 ? tanh_abs : -tanh_abs;
    int F_Q = (tanh_z * 10 * 256) / 256;"""


# ─── Property A ──────────────────────────────────────────────────────────────

def harness_prop_a_right(ctrl_body):
    return f"""\
/*
 * ddpg_prop_a_right.c — Property A: θ > {DANGER_TH}/256 rad, θ̇ ≥ 0 → F > 0
 *
 * Usa monotonicidade de tanh: z > 0 ⟺ tanh(z) > 0 ⟺ F > 0.
 * Não precisa aproximar tanh.
 *
 * ESBMC FAILED = controlador aplica força na direção errada
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
    __ESBMC_assume(th  >  {DANGER_TH} && th  <= {TH_BND});
    __ESBMC_assume(thd >= 0           && thd <= {THD_BND});

{ctrl_body}

    /* z > 0 ⟹ tanh(z) > 0 ⟹ F > 0 (monotonicidade) */
    __ESBMC_assert(z > 0, "PropA-right: controlador nao aplica forca positiva!");
    return 0;
}}
"""


def harness_prop_a_left(ctrl_body):
    return f"""\
/*
 * ddpg_prop_a_left.c — Property A: θ < -{DANGER_TH}/256 rad, θ̇ ≤ 0 → F < 0
 *
 * ESBMC FAILED = controlador aplica força na direção errada
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
    __ESBMC_assume(th  < -{DANGER_TH} && th  >= -{TH_BND});
    __ESBMC_assume(thd <= 0            && thd >= -{THD_BND});

{ctrl_body}

    __ESBMC_assert(z < 0, "PropA-left: controlador nao aplica forca negativa!");
    return 0;
}}
"""


# ─── Property B ──────────────────────────────────────────────────────────────

def harness_prop_b(ctrl_body):
    return f"""\
/*
 * ddpg_prop_b_safety.c — Property B: segurança em 1 passo (dinâmica linearizada)
 *
 * Usa aproximação linear de tanh (MESMA do browser) para computar F.
 * Dinâmica linearizada: sin(θ)≈θ, cos(θ)≈1.
 *
 * th_acc = (4040 * th - 375 * F_Q) / 256
 * th_new = th + (5 * thd) / 256
 * thd_new = thd + (5 * th_acc) / 256
 *
 * ESBMC FAILED = sistema sai da região segura em 1 passo
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
    __ESBMC_assume(th  >= -{TH_BND} && th  <= {TH_BND});
    __ESBMC_assume(thd >= -{THD_BND}&& thd <= {THD_BND});

{ctrl_body}

{TANH_APPROX_C}

    /* Dinâmica linearizada Q8.8 */
    int th_acc  = (4040 * th - 375 * F_Q) / 256;
    int th_new  = th  + (5 * thd) / 256;
    int thd_new = thd + (5 * th_acc) / 256;

    __ESBMC_assert(th_new >= -{TH_BND} && th_new <= {TH_BND},
                   "PropB: theta sai da regiao segura apos 1 passo!");
    return 0;
}}
"""


# ─── Property C ──────────────────────────────────────────────────────────────

def harness_prop_c(ctrl_body):
    return f"""\
/*
 * ddpg_prop_c_bounds.c — Property C: |F| ≤ 10 N sempre
 *
 * Verifica que a saída quantizada nunca excede os limites de força.
 * Deve ser SUCCESSFUL (tanh garante |output| ≤ 1 → |F| ≤ 10).
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
    __ESBMC_assume(th  >= -{TH_BND} && th  <= {TH_BND});
    __ESBMC_assume(thd >= -{THD_BND}&& thd <= {THD_BND});

{ctrl_body}

{TANH_APPROX_C}

    /* F_Q em Q8.8: 10N = 2560, -10N = -2560 */
    __ESBMC_assert(F_Q >= -2560 && F_Q <= 2560,
                   "PropC: forca excede limites [-10, +10] N!");
    return 0;
}}
"""


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_esbmc(c_file, timeout=TIMEOUT):
    try:
        r = subprocess.run(
            [ESBMC, c_file, "--no-unwinding-assertions", "--boolector"],
            capture_output=True, text=True, timeout=timeout,
        )
        out = r.stdout + r.stderr

        if "VERIFICATION SUCCESSFUL" in out:
            return True, "", out
        elif "VERIFICATION FAILED" in out:
            ce_parts = []
            for name, label in [("x", "x"), ("xd", "xd"), ("th", "th"), ("thd", "thd")]:
                m = re.search(rf'\b{name}\s*=\s*(-?\d+)', out)
                if m:
                    val = int(m.group(1))
                    ce_parts.append(f"{label}={val / SCALE:.4f}")
            m_z = re.search(r'\bz\s*=\s*(-?\d+)', out)
            if m_z:
                ce_parts.append(f"z={m_z.group(1)}")
            m_f = re.search(r'\bF_Q\s*=\s*(-?\d+)', out)
            if m_f:
                ce_parts.append(f"F_Q={int(m_f.group(1)) / SCALE:.2f}N")
            return False, "  ".join(ce_parts) if ce_parts else "(ver saída ESBMC)", out
        else:
            return None, "resultado desconhecido", out
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT", ""


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Verificação em Malha Fechada — DDPG Actor")
    print(f"Domínio Q8.8 (scale={SCALE}): x∈[±{X_BND}] xd∈[±{XD_BND}]"
          f" th∈[±{TH_BND}] thd∈[±{THD_BND}]")
    print(f"Limiar de perigo (Property A): |θ| > {DANGER_TH}/256"
          f" ≈ {DANGER_TH / SCALE * 180 / math.pi:.1f}°")
    print("=" * 65)

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

    ctrl_body = generate_controller_body(
        qw1, qb1, qw2, qb2, qw_out, qb_out,
        lo_pre1, hi_pre1, lo_pre2, hi_pre2)

    results = {}
    properties = [
        ("property_a_right", "Property A — θ > threshold → F > 0",
         harness_prop_a_right(ctrl_body)),
        ("property_a_left", "Property A — θ < -threshold → F < 0",
         harness_prop_a_left(ctrl_body)),
        ("property_b_safety", "Property B — Segurança 1-step",
         harness_prop_b(ctrl_body)),
        ("property_c_bounds", "Property C — |F| ≤ 10 N",
         harness_prop_c(ctrl_body)),
    ]

    for key, desc, src in properties:
        print(f"\n{'─' * 65}")
        print(f"{desc}")
        print(f"{'─' * 65}")

        c_file = f"/tmp/ddpg_cl_{key}.c"
        with open(c_file, "w") as f:
            f.write(src)
        print(f"Harness: {c_file}")
        print(f"Executando ESBMC (timeout={TIMEOUT}s)...")

        ok, ce, raw = run_esbmc(c_file)

        if ok is True:
            print(f"\n  SUCCESSFUL — propriedade satisfeita")
            results[key] = {"result": "SUCCESSFUL", "counterexample": ""}
        elif ok is False:
            print(f"\n  FAILED — CONTRAEXEMPLO: {ce}")
            results[key] = {"result": "FAILED", "counterexample": ce}
        else:
            print(f"\n  {ce}")
            results[key] = {"result": ce, "counterexample": ""}

    # Resumo
    print(f"\n{'=' * 65}")
    print("RESUMO — Verificação Malha Fechada DDPG")
    print(f"{'=' * 65}")
    for key, r in results.items():
        status = r["result"]
        ce = r.get("counterexample", "")
        print(f"\n  {key}: {status}")
        if ce:
            print(f"    Contraexemplo: {ce}")

    out_file = os.path.join(HERE, "ddpg_closed_loop_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {out_file}")

    return results


if __name__ == "__main__":
    main()
