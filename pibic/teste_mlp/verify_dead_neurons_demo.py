"""
verify_dead_neurons_demo.py — Demonstra verificação de neurônio morto com ESBMC.

Modifica artificialmente o viés do neurônio 1 para b=-1000 (Q8.8),
garantindo que pre <= 0 para todo (x1,x2) em [0,256]²:

  max(pre_1) = (0*(-750))/256 + (256*750)/256 + (-1000) = 750 - 1000 = -250 <= 0

Resultado esperado:
  Neurônio 0: VIVO   (ESBMC: VERIFICATION FAILED — encontrou input que o ativa)
  Neurônio 1: MORTO  (ESBMC: VERIFICATION SUCCESSFUL — provado que nunca ativa)
  Neurônio 2: VIVO
  Neurônio 3: VIVO
"""

import subprocess
import re
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from onnx_mlp_extractor import extract_mlp_weights

ESBMC = os.path.join(os.path.dirname(__file__),
                     "../QNNVerifier/esbmc-6.8.0/esbmc")

SCALE = 256
ONNX_FILE = os.path.join(os.path.dirname(__file__), "mlp_model.onnx")


def q(v):
    return int(round(v * SCALE))


def generate_harness(neuron_idx, qw_hidden, qb_hidden, q_bound):
    i = neuron_idx
    return f"""\
/*
 * verify_dead_neuron_demo_{i}.c — pesos originais com b[1] = -1000
 *
 * VERIFICATION FAILED  = neuronio VIVO  (existe entrada que o ativa)
 * VERIFICATION SUCCESS = neuronio MORTO (nunca ativa para nenhuma entrada)
 */
void __ESBMC_assume(_Bool cond);
void __ESBMC_assert(_Bool cond, const char *msg);
int nondet_int(void);

int main(void) {{
    int x1 = nondet_int();
    int x2 = nondet_int();
    __ESBMC_assume(x1 >= 0 && x1 <= {SCALE});
    __ESBMC_assume(x2 >= 0 && x2 <= {SCALE});

    int pre = (x1 * ({qw_hidden[i][0]})) / {SCALE}
            + (x2 * ({qw_hidden[i][1]})) / {SCALE}
            + ({qb_hidden[i]});

    __ESBMC_assume(pre >= -{q_bound} && pre <= {q_bound});

    int h = pre > 0 ? pre : 0;

    __ESBMC_assert(h == 0, "neuronio {i} e sempre morto?");
    return 0;
}}
"""


def run_esbmc(c_file):
    result = subprocess.run(
        [ESBMC, c_file, "--no-unwinding-assertions", "--boolector"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    succeeded = "VERIFICATION SUCCESSFUL" in output
    cx1 = re.search(r'\bx1\s*=\s*(-?\d+)', output)
    cx2 = re.search(r'\bx2\s*=\s*(-?\d+)', output)
    ch  = re.search(r'\bh\s*=\s*(-?\d+)', output)
    ce = ""
    if cx1 and cx2:
        vx1, vx2 = int(cx1.group(1)), int(cx2.group(1))
        hval = int(ch.group(1)) if ch else "?"
        ce = (f"x1={vx1}(={vx1/SCALE:.3f}), "
              f"x2={vx2}(={vx2/SCALE:.3f}), h={hval}")
    return succeeded, ce


def main():
    weights  = extract_mlp_weights(ONNX_FILE)
    qw_hidden = [[q(v) for v in row] for row in weights["w_hidden"]]
    qb_hidden = [q(v) for v in weights["b_hidden"]]
    q_bound   = int(5.0 * SCALE)

    # ── Modificação: mata artificialmente o neurônio 1 ──────────────────────
    DEAD_NEURON   = 1
    ORIGINAL_BIAS = qb_hidden[DEAD_NEURON]
    DEAD_BIAS     = -1000          # garante max(pre)=750-1000=-250 <= 0

    print("=" * 62)
    print("Demonstração — Neurônio Morto Artificial (neurônio 1)")
    print("=" * 62)
    print(f"\nModificação: b[{DEAD_NEURON}]  {ORIGINAL_BIAS}  →  {DEAD_BIAS}")
    print(f"  max(pre) = 750 + ({DEAD_BIAS}) = {750 + DEAD_BIAS}  <= 0  =>  h sempre 0\n")

    qb_hidden[DEAD_NEURON] = DEAD_BIAS

    print(f"{'Neurônio':<10} {'Pesos':<20} {'Viés':>6}   {'Resultado ESBMC'}")
    print("-" * 62)

    results = []
    for i in range(4):
        tag = " ← MORTO" if i == DEAD_NEURON else ""
        harness = generate_harness(i, qw_hidden, qb_hidden, q_bound)
        c_file  = f"/tmp/verify_dead_demo_{i}.c"
        with open(c_file, "w") as f:
            f.write(harness)

        succeeded, ce = run_esbmc(c_file)

        if succeeded:
            status = f"\033[92mMORTO  — VERIFICATION SUCCESSFUL\033[0m"
        else:
            status = f"\033[91mVIVO   — VERIFICATION FAILED\033[0m"
            if ce:
                status += f"\n{'':38}   ({ce})"

        print(f"  h[{i}]{tag:<14} {str(qw_hidden[i]):<20} {qb_hidden[i]:>6}   {status}")
        results.append((i, succeeded))

    print("\n" + "=" * 62)
    dead  = [r[0] for r in results if r[1]]
    alive = [r[0] for r in results if not r[1]]
    print(f"Neurônios MORTOS  (provados): {dead}")
    print(f"Neurônios VIVOS   (provados): {alive}")
    print()
    if dead == [DEAD_NEURON]:
        print("✓ Demonstração bem-sucedida: ESBMC detectou exatamente o")
        print(f"  neurônio modificado ({DEAD_NEURON}) como morto e os demais como vivos.")


if __name__ == "__main__":
    main()
