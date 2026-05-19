"""
verify_dead_neurons.py — Verifica formalmente se algum neurônio oculto do MLP XOR
é sempre inativo (neurônio morto) para qualquer entrada em [0, 256]².

Fluxo:
  1. Lê pesos do mlp_model.onnx via onnx_mlp_extractor
  2. Quantiza os pesos (scale=256)
  3. Para cada neurônio i (0..3):
       - Gera harness C com entradas simbólicas (nondet_int)
       - __ESBMC_assert(h[i] == 0, "neurônio i sempre morto?")
       - Roda ESBMC/Boolector
       - FAILED  → neurônio VIVO (mostra contraexemplo: qual entrada o ativa)
       - SUCCESS → neurônio MORTO (nunca ativa para nenhuma entrada válida)
  4. Imprime relatório final

Uso:
    python verify_dead_neurons.py
"""

import subprocess
import re
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from onnx_mlp_extractor import extract_mlp_weights

ESBMC = os.path.join(os.path.dirname(__file__),
                     "../../pibic/QNNVerifier/esbmc-6.8.0/esbmc")
if not os.path.isfile(ESBMC):
    ESBMC = os.path.join(os.path.dirname(__file__),
                         "../QNNVerifier/esbmc-6.8.0/esbmc")

SCALE = 256
ONNX_FILE = os.path.join(os.path.dirname(__file__), "mlp_model.onnx")


def q(v):
    return int(round(v * SCALE))


def generate_harness(neuron_idx, qw_hidden, qb_hidden, q_bound):
    """Gera harness C que verifica se o neurônio neuron_idx é sempre morto."""
    i = neuron_idx
    return f"""\
/*
 * verify_dead_neuron_{i}.c
 *
 * Verifica se o neurônio oculto {i} do MLP XOR é sempre inativo
 * para qualquer entrada (x1, x2) no domínio [0, 256]² (Q8.8).
 *
 * VERIFICATION FAILED  → neurônio VIVO (contraexemplo = entrada que o ativa)
 * VERIFICATION SUCCESS → neurônio MORTO (nunca ativa)
 */

void __ESBMC_assume(_Bool cond);
void __ESBMC_assert(_Bool cond, const char *msg);
int nondet_int(void);

static int relu_int(int x) {{ return x > 0 ? x : 0; }}

int main(void) {{
    /* entradas simbolicas — representam QUALQUER valor em [0,1] (Q8.8) */
    int x1 = nondet_int();
    int x2 = nondet_int();
    __ESBMC_assume(x1 >= 0 && x1 <= {SCALE});
    __ESBMC_assume(x2 >= 0 && x2 <= {SCALE});

    /* pre-ativacao do neuronio {i} */
    int pre = (x1 * ({qw_hidden[i][0]})) / {SCALE}
            + (x2 * ({qw_hidden[i][1]})) / {SCALE}
            + ({qb_hidden[i]});

    /* invariante conservadora pos-ReLU */
    __ESBMC_assume(pre >= -{q_bound} && pre <= {q_bound});

    int h = relu_int(pre);

    /* FAILED = neuronio VIVO (existe entrada que o ativa)
       SUCCESS = neuronio MORTO (sempre zero)               */
    __ESBMC_assert(h == 0, "neuronio {i} e sempre morto?");

    return 0;
}}
"""


def run_esbmc(c_file):
    """Roda ESBMC e retorna (succeeded, counterexample_str)."""
    result = subprocess.run(
        [ESBMC, c_file, "--no-unwinding-assertions", "--boolector"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    succeeded = "VERIFICATION SUCCESSFUL" in output
    # Extrai valores do contraexemplo (x1, x2)
    cx1 = re.search(r'\bx1\s*=\s*(-?\d+)', output)
    cx2 = re.search(r'\bx2\s*=\s*(-?\d+)', output)
    ce = ""
    if cx1 and cx2:
        vx1 = int(cx1.group(1))
        vx2 = int(cx2.group(1))
        ce = (f"x1={vx1} (≈{vx1/SCALE:.3f}), "
              f"x2={vx2} (≈{vx2/SCALE:.3f})")
    return succeeded, ce, output


def main():
    # Carrega e quantiza pesos
    weights = extract_mlp_weights(ONNX_FILE)
    qw_hidden = [[q(v) for v in row] for row in weights["w_hidden"]]
    qb_hidden = [q(v) for v in weights["b_hidden"]]
    q_bound = int(5.0 * SCALE)

    print("=" * 60)
    print("Verificação de Neurônios Mortos — MLP XOR (2→4→1)")
    print(f"Domínio de entrada: x1,x2 ∈ [0,{SCALE}] (Q8.8 ≡ [0,1])")
    print("=" * 60)

    results = []
    for i in range(4):
        harness = generate_harness(i, qw_hidden, qb_hidden, q_bound)
        c_file = f"/tmp/verify_dead_neuron_{i}.c"
        with open(c_file, "w") as f:
            f.write(harness)

        print(f"\n[Neurônio {i}] w={qw_hidden[i]}, b={qb_hidden[i]}")
        print(f"  Rodando ESBMC...", end=" ", flush=True)

        try:
            succeeded, ce, raw = run_esbmc(c_file)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            results.append((i, "timeout", ""))
            continue

        if succeeded:
            print("MORTO ✗  (nunca ativa para nenhuma entrada)")
            results.append((i, "dead", ""))
        else:
            print(f"VIVO ✓  ({ce if ce else 'neurônio ativa para alguma entrada'})")
            results.append((i, "alive", ce))

    # Relatório final
    print("\n" + "=" * 60)
    print("RELATÓRIO FINAL")
    print("=" * 60)
    dead = [r for r in results if r[1] == "dead"]
    alive = [r for r in results if r[1] == "alive"]

    if dead:
        print(f"\n⚠  Neurônios MORTOS (candidatos a poda): {[r[0] for r in dead]}")
        print("   Estes pesos nunca contribuem para a saída — modelo pode ser podado.")
    else:
        print("\n✓  Nenhum neurônio morto: todos os 4 neurônios contribuem para a saída.")

    if alive:
        print(f"\n✓  Neurônios VIVOS: {[r[0] for r in alive]}")
        for r in alive:
            if r[2]:
                print(f"   Neurônio {r[0]}: {r[2]}")

    print()


if __name__ == "__main__":
    main()
