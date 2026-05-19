"""
verify_closed_loop.py — Verificação formal em malha fechada do controlador DQN.

Propriedades verificadas:

  Property A — Detecção de "direção errada" (sem dinâmica, tratável):
    Para qualquer estado onde θ > threshold AND θ_dot ≥ 0:
      DQN deve emitir ação=1 (empurrar à direita para restaurar equilíbrio)
      Se emitir ação=0, é falha do controlador.
    Simétrico: θ < -threshold AND θ_dot ≤ 0 → assert(action==0)

  Property B — Segurança em passo único com dinâmica linearizada:
    Para s₀ ∈ S_safe (simbólico e limitado):
      1. Executa passagem completa do DQN para obter ação
      2. Aplica UM passo de dinâmica linearizada (sin≈θ, cos≈1)
      3. Verifica que θ₁ ∈ [-53, 53] (em Q8.8)

  ESBMC FAILED   = contraexemplo encontrado (falha detectada)
  ESBMC SUCCESS  = propriedade satisfeita para todo estado no domínio

Uso:
    python verify_closed_loop.py
"""

import subprocess, re, os, sys, json, math, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_controller_extractor import extract_controller_weights

ESBMC     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../QNNVerifier/esbmc-6.8.0/esbmc")
SCALE     = 256
ONNX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "dqn_cartpole.onnx")
TIMEOUT   = 120  # segundos por propriedade

# Bounds do estado em Q8.8
X_BND   = int(2.4   * SCALE)    # 614
XD_BND  = int(5.0   * SCALE)    # 1280
TH_BND  = int(0.2094 * SCALE)   # 53
THD_BND = int(5.0   * SCALE)    # 1280

# Limiar de "zona de perigo" para Property A: ~5.7° em Q8.8
DANGER_TH = int(0.10 * SCALE)   # 25  (≈ 5.7°)


def q(v):
    return int(round(v * SCALE))


def c_div(a, b):
    """Divisão inteira truncando em direção a zero — igual ao '/' do C."""
    return int(a / b)


# ─── Propagação de intervalo ──────────────────────────────────────────────────

def compute_pre_bounds(qw, qb, in_lo, in_hi):
    """Aritmética de intervalo para bounds de pré-ativação.

    Usa c_div() que trunca em direção a zero, exatamente como o operador '/'
    do harness C gerado. Usar Python '//' (floor) produziria bounds menores
    para termos negativos, tornando os __ESBMC_assume muito restritivos e
    a prova não-soa.
    """
    lo_pre, hi_pre = [], []
    for i in range(len(qb)):
        lo = qb[i]
        hi = qb[i]
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


def compute_h_bounds(lo_pre, hi_pre):
    """Aplica ReLU nos bounds de pré-ativação."""
    return [max(0, lo) for lo in lo_pre], [max(0, hi) for hi in hi_pre]


# ─── Geração do harness C ─────────────────────────────────────────────────────

def generate_controller_harness(
        qw1, qb1, qw2, qb2, qw_out, qb_out,
        lo_pre1, hi_pre1, lo_pre2, hi_pre2):
    """
    Gera o trecho C com o controlador DQN completamente expandido (sem loops).
    Retorna a string do bloco da função controller().
    """
    lines = []

    # ── Pesos camada 1 ──────────────────────────────────────────────────────
    # (embutidos como constantes no próprio código para evitar arrays globais)

    # Camada 1: 24 neurônios × 4 entradas
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

    # Camada 2: 24 neurônios × 24 entradas
    for j in range(24):
        terms = " + ".join(f"(h1_{k}*({qw2[j][k]}))/256" for k in range(24))
        b = qb2[j]
        lines.append(f"    int pre2_{j} = {terms} + ({b});")
        lines.append(
            f"    __ESBMC_assume(pre2_{j} >= {lo_pre2[j]} && pre2_{j} <= {hi_pre2[j]});"
        )
        lines.append(f"    int h2_{j} = pre2_{j} > 0 ? pre2_{j} : 0;")

    lines.append("")

    # Camada de saída: 2 neurônios × 24 entradas
    q0_terms = " + ".join(f"(h2_{k}*({qw_out[0][k]}))/256" for k in range(24))
    q1_terms = " + ".join(f"(h2_{k}*({qw_out[1][k]}))/256" for k in range(24))
    lines.append(f"    int q0 = {q0_terms} + ({qb_out[0]});")
    lines.append(f"    int q1 = {q1_terms} + ({qb_out[1]});")
    lines.append("    int action = (q1 > q0) ? 1 : 0;")

    return "\n".join(lines)


# ─── Property A ───────────────────────────────────────────────────────────────

def harness_prop_a_right(controller_body):
    """
    θ > DANGER_TH AND θ_dot ≥ 0 → controlador deve empurrar à direita (ação=1).
    ESBMC FAILED = encontrou estado onde empurra à esquerda (FALHA!).
    """
    return f"""\
/*
 * prop_a_right.c — Property A (direção errada, lado direito)
 *
 * Quando θ > {DANGER_TH}/256 ≈ {DANGER_TH/256:.3f} rad ({DANGER_TH/256*180/math.pi:.1f}°)
 * E θ_dot ≥ 0 (pêndulo inclinando à direita):
 *   Controlador DEVE escolher ação=1 (empurrar à direita).
 *   Se escolher ação=0, é uma falha de controle.
 *
 * ESBMC FAILED   = contraexemplo: estado onde controlador empurra na direção errada
 * ESBMC SUCCESS  = controlador nunca empurra na direção errada nesta zona
 */
void __ESBMC_assume(_Bool c);
void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
    /* Estado simbólico em Q8.8 — zona de perigo: pêndulo inclinando para direita */
    int x   = nondet_int();
    int xd  = nondet_int();
    int th  = nondet_int();
    int thd = nondet_int();

    __ESBMC_assume(x   >= -{X_BND}  && x   <= {X_BND});
    __ESBMC_assume(xd  >= -{XD_BND} && xd  <= {XD_BND});
    /* zona de perigo: θ > threshold E θ_dot ≥ 0 */
    __ESBMC_assume(th  >  {DANGER_TH} && th  <= {TH_BND});
    __ESBMC_assume(thd >= 0           && thd <= {THD_BND});

    /* Passagem completa do DQN (pesos quantizados Q8.8) */
{controller_body}

    /* Propriedade: controlador deve empurrar à direita (action==1) */
    __ESBMC_assert(action == 1, "PropA-right: controlador empurra na direcao errada!");
    return 0;
}}
"""


def harness_prop_a_left(controller_body):
    """
    θ < -DANGER_TH AND θ_dot ≤ 0 → controlador deve empurrar à esquerda (ação=0).
    ESBMC FAILED = encontrou estado onde empurra à direita (FALHA!).
    """
    return f"""\
/*
 * prop_a_left.c — Property A (direção errada, lado esquerdo)
 *
 * Quando θ < -{DANGER_TH}/256 ≈ -{DANGER_TH/256:.3f} rad (inclinando à esquerda)
 * E θ_dot ≤ 0 (pêndulo inclinando mais à esquerda):
 *   Controlador DEVE escolher ação=0 (empurrar à esquerda).
 *
 * ESBMC FAILED   = contraexemplo: estado onde controlador empurra na direção errada
 * ESBMC SUCCESS  = controlador nunca empurra na direção errada nesta zona
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
    /* zona de perigo: θ < -threshold E θ_dot ≤ 0 */
    __ESBMC_assume(th  < -{DANGER_TH} && th  >= -{TH_BND});
    __ESBMC_assume(thd <= 0            && thd >= -{THD_BND});

    /* Passagem completa do DQN */
{controller_body}

    /* Propriedade: controlador deve empurrar à esquerda (action==0) */
    __ESBMC_assert(action == 0, "PropA-left: controlador empurra na direcao errada!");
    return 0;
}}
"""


# ─── Property B ───────────────────────────────────────────────────────────────

def harness_prop_b(controller_body):
    """
    Segurança em passo único com dinâmica linearizada (sin≈θ, cos≈1).

    Fórmulas Q8.8 derivadas dos parâmetros físicos:
      Parâmetros: g=9.8, M=1.0, m=0.1, L=0.5, F_mag=10N, dt=0.02
      ML = m*L = 0.05
      M_total = 1.1
      Denominador para th_acc ≈ L*(4/3 - m/M_total) = 0.5*(1.333 - 0.0909) ≈ 0.621
      Com linearização (sin≈θ, cos≈1, xd²≈0):
        temp = (F + ML*thd²*sin_th) / M_total ≈ F/M_total  (ignorando segundo termo)
        th_acc ≈ (g*th - temp) / 0.621
               ≈ (9.8*th - F/1.1) / 0.621
      Em Q8.8 (multiplicar por 256):
        th_acc_Q ≈ (9.8*256/0.621 * th_Q/256 - (F_Q/256)/1.1) / 0.621...

      Simplificado diretamente:
        th_acc_Q = (4040 * th_Q - 375 * F_Q) / 256
          onde: F_Q = 2560 (+10N) ou -2560 (-10N)

      Integração de Euler Q8.8:
        th_new  = th + (5 * thd) / 256     [dt*256 = 0.02*256 ≈ 5]
        thd_new = thd + (5 * th_acc_Q) / 256
    """
    return f"""\
/*
 * prop_b_safety.c — Property B: segurança em passo único (dinâmica linearizada)
 *
 * Para qualquer s0 ∈ S_safe:
 *   1. Executa DQN para obter ação
 *   2. Aplica UM passo de dinâmica linearizada (sin≈θ, cos≈1)
 *   3. Verifica que th_new ∈ [-{TH_BND}, {TH_BND}] (seguro)
 *
 * ESBMC FAILED   = encontrou estado onde sistema sai da região segura em 1 passo
 * ESBMC SUCCESS  = sistema sempre permanece seguro após 1 passo
 *
 * Linearização (válida para |θ| ≤ 12°):
 *   th_acc_Q = (4040 * th - 375 * F_Q) / 256
 *   th_new   = th + (5 * thd) / 256
 *   thd_new  = thd + (5 * th_acc_Q) / 256
 *   F_Q = 2560 se action==1, -2560 se action==0
 */
void __ESBMC_assume(_Bool c);
void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
    /* Estado simbólico s0 ∈ S_safe em Q8.8 */
    int x   = nondet_int();
    int xd  = nondet_int();
    int th  = nondet_int();
    int thd = nondet_int();

    __ESBMC_assume(x   >= -{X_BND}  && x   <= {X_BND});
    __ESBMC_assume(xd  >= -{XD_BND} && xd  <= {XD_BND});
    __ESBMC_assume(th  >= -{TH_BND} && th  <= {TH_BND});
    __ESBMC_assume(thd >= -{THD_BND}&& thd <= {THD_BND});

    /* Passagem completa do DQN */
{controller_body}

    /* Dinâmica linearizada Q8.8 */
    int F_Q     = (action == 1) ? 2560 : -2560;
    int th_acc  = (4040 * th - 375 * F_Q) / 256;
    int th_new  = th  + (5 * thd)   / 256;
    int thd_new = thd + (5 * th_acc) / 256;

    /* Propriedade: pêndulo deve permanecer na região segura após 1 passo */
    __ESBMC_assert(th_new >= -{TH_BND} && th_new <= {TH_BND},
                   "PropB: theta sai da regiao segura apos 1 passo!");
    return 0;
}}
"""


# ─── Runner ESBMC ─────────────────────────────────────────────────────────────

def run_esbmc(c_file, timeout=TIMEOUT):
    """Executa ESBMC e retorna (ok, counterexample_str, raw_output)."""
    try:
        r = subprocess.run(
            [ESBMC, c_file, "--no-unwinding-assertions", "--boolector"],
            capture_output=True, text=True, timeout=timeout,
        )
        out = r.stdout + r.stderr

        if "VERIFICATION SUCCESSFUL" in out:
            return True, "", out
        elif "VERIFICATION FAILED" in out:
            # Tenta extrair valores do contraexemplo
            ce_parts = []
            for name, label in [("x", "x"), ("xd", "xd"), ("th", "th"), ("thd", "thd")]:
                m = re.search(rf'\b{name}\s*=\s*(-?\d+)', out)
                if m:
                    val = int(m.group(1))
                    ce_parts.append(f"{label}={val/SCALE:.4f}")
            # Também tenta action
            m_act = re.search(r'\baction\s*=\s*(-?\d+)', out)
            if m_act:
                ce_parts.append(f"action={m_act.group(1)}")
            ce_str = "  ".join(ce_parts) if ce_parts else "(ver saída ESBMC)"
            return False, ce_str, out
        else:
            return None, "resultado desconhecido", out
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT", ""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Verificação em Malha Fechada — DQN Cart-Pole")
    print(f"Domínio Q8.8 (scale={SCALE}): x∈[±{X_BND}] xd∈[±{XD_BND}]"
          f" th∈[±{TH_BND}] thd∈[±{THD_BND}]")
    print(f"Limiar de perigo (Property A): |θ| > {DANGER_TH}/256"
          f" ≈ {DANGER_TH/256*180/math.pi:.1f}°")
    print("=" * 65)

    # ── Carrega e quantiza pesos ──────────────────────────────────────────────
    print("\nCarregando pesos do ONNX...")
    weights = extract_controller_weights(ONNX_FILE)

    qw1   = [[q(v) for v in row] for row in weights["w1"]]
    qb1   =  [q(v) for v in weights["b1"]]
    qw2   = [[q(v) for v in row] for row in weights["w2"]]
    qb2   =  [q(v) for v in weights["b2"]]
    qw_out= [[q(v) for v in row] for row in weights["w_out"]]
    qb_out=  [q(v) for v in weights["b_out"]]
    print("Pesos carregados e quantizados.")

    # ── Calcula bounds analíticos ─────────────────────────────────────────────
    in_lo = [-X_BND, -XD_BND, -TH_BND, -THD_BND]
    in_hi = [ X_BND,  XD_BND,  TH_BND,  THD_BND]

    lo_pre1, hi_pre1 = compute_pre_bounds(qw1, qb1, in_lo, in_hi)
    lo_h1,   hi_h1   = compute_h_bounds(lo_pre1, hi_pre1)
    lo_pre2, hi_pre2 = compute_pre_bounds(qw2, qb2, lo_h1, hi_h1)

    print("Bounds de pré-ativação calculados (aritmética de intervalo).")

    # ── Gera corpo do controlador ─────────────────────────────────────────────
    ctrl_body = generate_controller_harness(
        qw1, qb1, qw2, qb2, qw_out, qb_out,
        lo_pre1, hi_pre1, lo_pre2, hi_pre2
    )

    results = {}

    # ── Property A (lado direito) ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"Property A — θ > {DANGER_TH/256:.2f} rad AND θ_dot ≥ 0 → push right (action=1)")
    print(f"{'─'*65}")
    src_ar = harness_prop_a_right(ctrl_body)
    f_ar = "/tmp/cl_prop_a_right.c"
    with open(f_ar, "w") as f:
        f.write(src_ar)
    print(f"Harness gerado: {f_ar}")
    print(f"Executando ESBMC (timeout={TIMEOUT}s)...")

    ok_ar, ce_ar, out_ar = run_esbmc(f_ar)

    if ok_ar is True:
        print("\nProperty A (direita): SUCCESSFUL")
        print("  Controlador NUNCA empurra na direção errada nesta zona")
        results["property_a_right"] = {"result": "SUCCESSFUL", "counterexample": ""}
    elif ok_ar is False:
        print("\nProperty A (direita): FAILED — CONTRAEXEMPLO ENCONTRADO!")
        print(f"  Estado: {ce_ar}")
        print("  CONTROLADOR EMPURRA NA DIRECAO ERRADA!")
        results["property_a_right"] = {"result": "FAILED", "counterexample": ce_ar}
    else:
        print(f"\nProperty A (direita): {ce_ar}")
        results["property_a_right"] = {"result": ce_ar, "counterexample": ""}

    # ── Property A (lado esquerdo) ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"Property A — θ < -{DANGER_TH/256:.2f} rad AND θ_dot ≤ 0 → push left (action=0)")
    print(f"{'─'*65}")
    src_al = harness_prop_a_left(ctrl_body)
    f_al = "/tmp/cl_prop_a_left.c"
    with open(f_al, "w") as f:
        f.write(src_al)
    print(f"Harness gerado: {f_al}")
    print(f"Executando ESBMC (timeout={TIMEOUT}s)...")

    ok_al, ce_al, out_al = run_esbmc(f_al)

    if ok_al is True:
        print("\nProperty A (esquerda): SUCCESSFUL")
        print("  Controlador NUNCA empurra na direção errada nesta zona")
        results["property_a_left"] = {"result": "SUCCESSFUL", "counterexample": ""}
    elif ok_al is False:
        print("\nProperty A (esquerda): FAILED — CONTRAEXEMPLO ENCONTRADO!")
        print(f"  Estado: {ce_al}")
        print("  CONTROLADOR EMPURRA NA DIRECAO ERRADA!")
        results["property_a_left"] = {"result": "FAILED", "counterexample": ce_al}
    else:
        print(f"\nProperty A (esquerda): {ce_al}")
        results["property_a_left"] = {"result": ce_al, "counterexample": ""}

    # ── Property B ────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Property B — Segurança em passo único (dinâmica linearizada)")
    print(f"{'─'*65}")
    src_b = harness_prop_b(ctrl_body)
    f_b = "/tmp/cl_prop_b_safety.c"
    with open(f_b, "w") as f:
        f.write(src_b)
    print(f"Harness gerado: {f_b}")
    print(f"Executando ESBMC (timeout={TIMEOUT}s)...")

    ok_b, ce_b, out_b = run_esbmc(f_b)

    if ok_b is True:
        print("\nProperty B (segurança 1 passo): SUCCESSFUL")
        print("  Sistema SEMPRE permanece seguro após 1 passo para todo s0 ∈ S_safe")
        results["property_b_safety"] = {"result": "SUCCESSFUL", "counterexample": ""}
    elif ok_b is False:
        print("\nProperty B (segurança 1 passo): FAILED — CONTRAEXEMPLO ENCONTRADO!")
        print(f"  Estado s0: {ce_b}")
        print("  SISTEMA SAI DA REGIAO SEGURA APOS 1 PASSO!")
        results["property_b_safety"] = {"result": "FAILED", "counterexample": ce_b}
    else:
        print(f"\nProperty B (segurança 1 passo): {ce_b}")
        results["property_b_safety"] = {"result": ce_b, "counterexample": ""}

    # ── Resumo final ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("RESUMO — Verificação em Malha Fechada")
    print(f"{'='*65}")
    print(f"\nProperty A (θ > {DANGER_TH/256:.2f} rad → push right):")
    r_ar = results["property_a_right"]
    if r_ar["result"] == "FAILED":
        print(f"  FAILED — contraexemplo: {r_ar['counterexample']} → ação=0 (CONTROLADOR FALHA!)")
    elif r_ar["result"] == "SUCCESSFUL":
        print("  SUCCESSFUL — nunca empurra na direção errada")
    else:
        print(f"  {r_ar['result']}")

    print(f"\nProperty A (θ < -{DANGER_TH/256:.2f} rad → push left):")
    r_al = results["property_a_left"]
    if r_al["result"] == "FAILED":
        print(f"  FAILED — contraexemplo: {r_al['counterexample']} → ação=1 (CONTROLADOR FALHA!)")
    elif r_al["result"] == "SUCCESSFUL":
        print("  SUCCESSFUL — nunca empurra na direção errada")
    else:
        print(f"  {r_al['result']}")

    print("\nProperty B (segurança em 1 passo):")
    r_b = results["property_b_safety"]
    if r_b["result"] == "FAILED":
        print(f"  FAILED — s₀ encontrado: {r_b['counterexample']} → θ₁ > limite")
    elif r_b["result"] == "SUCCESSFUL":
        print("  SUCCESSFUL — sistema seguro por 1 passo")
    else:
        print(f"  {r_b['result']}")

    print()

    # ── Salva resultados em JSON ──────────────────────────────────────────────
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "closed_loop_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Resultados salvos em: {out_file}")

    return results


if __name__ == "__main__":
    main()
