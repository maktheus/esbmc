"""
quantize_mlp.py — Lê pesos do MLP (via ONNX ou mlp_weights.h), quantiza para
inteiros (scale=256) e gera verify_mlp_qnn.c pronto para ESBMC.

Pipeline completo:
  mlp_training.py  →  mlp_model.onnx  (ou mlp_weights.h como fallback)
                              ↓
                     quantize_mlp.py   ←── este script
                              ↓
                    verify_mlp_qnn.c
                              ↓
  esbmc verify_mlp_qnn.c --no-unwinding-assertions --boolector

Uso:
  python quantize_mlp.py                   # usa mlp_model.onnx (padrão)
  python quantize_mlp.py --source h        # usa mlp_weights.h
  python quantize_mlp.py --source onnx     # usa mlp_model.onnx
"""

import re
import sys
import argparse

ONNX_FILE    = "mlp_model.onnx"
WEIGHTS_FILE = "mlp_weights.h"
OUTPUT_FILE  = "verify_mlp_qnn.c"
SCALE        = 256          # Q8.8
HIDDEN_BOUND = 5.0

# ---------------------------------------------------------------------------
# 1. Carrega pesos — ONNX (padrão) ou .h (fallback)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=["onnx", "h"], default="onnx",
                    help="Fonte dos pesos: 'onnx' (padrão) ou 'h'")
args = parser.parse_args()

if args.source == "onnx":
    from onnx_mlp_extractor import extract_mlp_weights
    weights  = extract_mlp_weights(ONNX_FILE)
    w_hidden = weights["w_hidden"]   # list[list[float]] [4][2]
    b_hidden = weights["b_hidden"]   # list[float] [4]
    w_out    = weights["w_out"]      # list[float] [4]
    b_out    = weights["b_out"]      # float
    source_label = ONNX_FILE
else:
    def parse_floats(text):
        return [float(x.rstrip('f')) for x in re.findall(r'-?\d+\.\d+(?:e[+-]?\d+)?f?', text)]

    with open(WEIGHTS_FILE) as f:
        src = f.read()

    m = re.search(r'w_hidden\[4\]\[2\]\s*=\s*\{([^;]+)\}', src, re.DOTALL)
    vals = parse_floats(m.group(1))
    w_hidden = [[vals[i*2], vals[i*2+1]] for i in range(4)]

    m = re.search(r'b_hidden\[4\]\s*=\s*\{([^;]+)\}', src, re.DOTALL)
    b_hidden = parse_floats(m.group(1))

    m = re.search(r'w_out\[4\]\s*=\s*\{([^;]+)\}', src, re.DOTALL)
    w_out = parse_floats(m.group(1))

    m = re.search(r'b_out\s*=\s*(-?\d+\.\d+(?:e[+-]?\d+)?f?)', src)
    b_out = float(m.group(1).rstrip('f'))
    source_label = WEIGHTS_FILE

print(f"=== Pesos lidos de {source_label} ===")
for i, row in enumerate(w_hidden):
    print(f"  w_hidden[{i}] = {row}")
print(f"  b_hidden   = {b_hidden}")
print(f"  w_out      = {w_out}")
print(f"  b_out      = {b_out}")

# ---------------------------------------------------------------------------
# 2. Quantiza para inteiros (arredondamento simétrico)
# ---------------------------------------------------------------------------

def q(v):
    return int(round(v * SCALE))

qw_hidden = [[q(v) for v in row] for row in w_hidden]
qb_hidden  = [q(v) for v in b_hidden]
qw_out     = [q(v) for v in w_out]
qb_out     = q(b_out)
q_bound    = int(HIDDEN_BOUND * SCALE)

print("\n=== Pesos quantizados (scale=256) ===")
for i, row in enumerate(qw_hidden):
    print(f"  qw_hidden[{i}] = {row}")
print(f"  qb_hidden  = {qb_hidden}")
print(f"  qw_out     = {qw_out}")
print(f"  qb_out     = {qb_out}")

# ---------------------------------------------------------------------------
# 3. Valida as 4 entradas XOR (simulação Python antes do ESBMC)
# ---------------------------------------------------------------------------

def relu(x): return max(0, x)

def mlp_float(x1, x2):
    hidden = [relu(x1*w_hidden[i][0] + x2*w_hidden[i][1] + b_hidden[i]) for i in range(4)]
    score  = sum(hidden[i]*w_out[i] for i in range(4)) + b_out
    return score

def c_trunc_div(a, b):
    """Integer division truncating toward zero — matches C's '/' on signed ints."""
    return int(a / b)

def mlp_qnn(x1, x2):
    hidden = []
    for i in range(4):
        h = (c_trunc_div(x1 * qw_hidden[i][0], SCALE)
             + c_trunc_div(x2 * qw_hidden[i][1], SCALE)
             + qb_hidden[i])
        hidden.append(max(0, h))
    score = qb_out
    for i in range(4):
        score += c_trunc_div(hidden[i] * qw_out[i], SCALE)
    return score

inputs_xor = [(0, 0, False), (0, SCALE, True), (SCALE, 0, True), (SCALE, SCALE, False)]

print("\n=== Simulação Python (validação pre-ESBMC) ===")
ok = True
for x1, x2, expect_true in inputs_xor:
    f_score  = mlp_float(x1/SCALE, x2/SCALE)
    q_score  = mlp_qnn(x1, x2)
    q_pass   = (q_score > 0) == expect_true
    status   = "OK" if q_pass else "FALHOU"
    print(f"  ({x1//SCALE},{x2//SCALE}): float={f_score:.3f}  quant={q_score}  esperado={'T' if expect_true else 'F'}  [{status}]")
    if not q_pass:
        ok = False

if not ok:
    print("\n[AVISO] Modelo quantizado nao satisfaz todas as propriedades XOR!")
    sys.exit(1)
else:
    print("\n[OK] Modelo quantizado satisfaz XOR para todas as 4 entradas.")

# ---------------------------------------------------------------------------
# 4. Gera verify_mlp_qnn.c
# ---------------------------------------------------------------------------

c = f"""\
/*
 * verify_mlp_qnn.c — Harness de verificacao formal (ESBMC)
 *
 * Gerado por quantize_mlp.py a partir de {source_label}
 *
 * Modelo: MLP 2->4->1, XOR
 * Quantizacao: inteiros puros, scale={SCALE} (Q8.8)
 *
 * Propriedades verificadas (4 casos concretos XOR):
 *   P1: mlp(0,0) <= 0   (XOR = false)
 *   P2: mlp(1,1) <= 0   (XOR = false)
 *   P3: mlp(0,1)  > 0   (XOR = true)
 *   P4: mlp(1,0)  > 0   (XOR = true)
 *
 * Verificar:
 *   esbmc verify_mlp_qnn.c --no-unwinding-assertions --boolector
 *
 * Esperado: VERIFICATION SUCCESSFUL
 */

#include <stdint.h>

void __ESBMC_assume(_Bool cond);
void __ESBMC_assert(_Bool cond, const char *msg);

/* Pesos quantizados — scale={SCALE} */
static int qw_hidden[4][2] = {{
    {{{qw_hidden[0][0]}, {qw_hidden[0][1]}}},
    {{{qw_hidden[1][0]}, {qw_hidden[1][1]}}},
    {{{qw_hidden[2][0]}, {qw_hidden[2][1]}}},
    {{{qw_hidden[3][0]}, {qw_hidden[3][1]}}}
}};
static int qb_hidden[4] = {{{qb_hidden[0]}, {qb_hidden[1]}, {qb_hidden[2]}, {qb_hidden[3]}}};
static int qw_out[4]    = {{{qw_out[0]}, {qw_out[1]}, {qw_out[2]}, {qw_out[3]}}};
static int qb_out       = {qb_out};

static int relu_int(int x) {{ return x > 0 ? x : 0; }}

static int mlp_forward(int x1, int x2) {{
    int h[4], i;
    for (i = 0; i < 4; i++) {{
        int pre = (x1 * qw_hidden[i][0]) / {SCALE}
                + (x2 * qw_hidden[i][1]) / {SCALE}
                + qb_hidden[i];
        __ESBMC_assume(pre >= -{q_bound} && pre <= {q_bound});
        h[i] = relu_int(pre);
    }}
    int score = qb_out;
    for (i = 0; i < 4; i++)
        score += (h[i] * qw_out[i]) / {SCALE};
    return score;
}}

int main(void) {{
    int zero = 0, um = {SCALE};

    /* P1 */ __ESBMC_assert(mlp_forward(zero, zero) <= 0, "P1: XOR(0,0) deve ser false");
    /* P2 */ __ESBMC_assert(mlp_forward(um,   um  ) <= 0, "P2: XOR(1,1) deve ser false");
    /* P3 */ __ESBMC_assert(mlp_forward(zero, um  )  > 0, "P3: XOR(0,1) deve ser true");
    /* P4 */ __ESBMC_assert(mlp_forward(um,   zero)  > 0, "P4: XOR(1,0) deve ser true");

    return 0;
}}
"""

with open(OUTPUT_FILE, 'w') as f:
    f.write(c)

print(f"\nGerado: {OUTPUT_FILE}")
print(f"Verificar com:")
print(f"  esbmc {OUTPUT_FILE} --no-unwinding-assertions --boolector")
