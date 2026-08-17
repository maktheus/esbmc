#!/usr/bin/env python3
"""
validate_forward.py — Prova que o forward em Verilog gerado por
gen_transition_system.py computa exatamente o mesmo z que a aritmética Q8.8 do
harness verificado (`cartpole/verify_ddpg_closed_loop.py:35-65`).

Sem isso, o resultado do IC3 seria sobre um controlador diferente do verificado,
e portanto não diria nada sobre o sistema real.

Método: emite um módulo puramente combinacional (sem registradores) com os
mesmos pesos, sintetiza com yosys, e usa `yosys eval` para avaliar z em estados
concretos — comparando contra a referência Python termo a termo.

Uso:
    python3 validate_forward.py            # 12 estados
    python3 validate_forward.py -n 400     # amostragem maior
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_transition_system import WEIGHTS, vconst, SCALE  # noqa: E402

X_BND, XD_BND, TH_BND, THD_BND = 614, 1280, 53, 1280


def cdiv(a: int, b: int) -> int:
    """Divisão inteira truncando para ZERO — o operador / do C."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def forward_ref(x, xd, th, thd, W):
    """Referência: mesma ordem de operações de generate_controller_body."""
    h1 = []
    for i in range(len(W["b1"])):
        a, b, c, d = W["w1"][i]
        pre = (cdiv(x * a, SCALE) + cdiv(xd * b, SCALE)
               + cdiv(th * c, SCALE) + cdiv(thd * d, SCALE) + W["b1"][i])
        h1.append(pre if pre > 0 else 0)
    h2 = []
    for j in range(len(W["b2"])):
        pre = sum(cdiv(h1[k] * W["w2"][j][k], SCALE)
                  for k in range(len(h1))) + W["b2"][j]
        h2.append(pre if pre > 0 else 0)
    z = sum(cdiv(h2[k] * W["w_out"][0][k], SCALE)
            for k in range(len(h2))) + W["b_out"][0]
    return z


def emit_comb(W) -> str:
    """Módulo combinacional: x,xd,th,thd -> z. Mesmas expressões do gerador."""
    H1, H2 = len(W["b1"]), len(W["b2"])
    L = ["module fwd(input signed [31:0] X, XD, TH, THD,",
         "            output signed [31:0] z);",
         "  function signed [31:0] td; input signed [31:0] v;",
         "    td = v[31] ? -((-v) >>> 8) : (v >>> 8); endfunction"]
    for i in range(H1):
        a, b, c, d = W["w1"][i]
        L.append(f"  wire signed [31:0] pre1_{i} = td({vconst(a)}*X)"
                 f" + td({vconst(b)}*XD) + td({vconst(c)}*TH)"
                 f" + td({vconst(d)}*THD) + {vconst(W['b1'][i])};")
        L.append(f"  wire signed [31:0] h1_{i} = pre1_{i}[31] ? 32'sd0 : pre1_{i};")
    for j in range(H2):
        t = " + ".join(f"td({vconst(W['w2'][j][k])}*h1_{k})" for k in range(H1))
        L.append(f"  wire signed [31:0] pre2_{j} = {t} + {vconst(W['b2'][j])};")
        L.append(f"  wire signed [31:0] h2_{j} = pre2_{j}[31] ? 32'sd0 : pre2_{j};")
    o = " + ".join(f"td({vconst(W['w_out'][0][k])}*h2_{k})" for k in range(H2))
    L.append(f"  assign z = {o} + {vconst(W['b_out'][0])};")
    L.append("endmodule")
    return "\n".join(L) + "\n"


def vlit(v: int, n: int = 32) -> str:
    """Literal para `yosys eval -set`. Uma string de bits nua seria lida como
    DECIMAL pelo yosys, silenciosamente — daí o prefixo de largura obrigatorio."""
    return f"{n}'d{v & ((1 << n) - 1)}"


def from_bits(s: str, n: int = 32) -> int:
    u = int(s, 2)
    return u - (1 << n) if u >> (n - 1) else u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=12, help="numero de estados a testar")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    with open(WEIGHTS) as fh:
        W = json.load(fh)

    comb = os.path.join(HERE, "_fwd_check.v")
    with open(comb, "w") as fh:
        fh.write(emit_comb(W))

    random.seed(a.seed)
    states = [(0, 0, 0, 0), (5, 5, 5, 5), (-5, -5, -5, -5),
              (X_BND, XD_BND, TH_BND, THD_BND),
              (-X_BND, -XD_BND, -TH_BND, -THD_BND)]
    while len(states) < a.n:
        states.append((random.randint(-X_BND, X_BND),
                       random.randint(-XD_BND, XD_BND),
                       random.randint(-TH_BND, TH_BND),
                       random.randint(-THD_BND, THD_BND)))
    states = states[:max(a.n, 5)]

    # um unico invocacao do yosys avalia todos os estados
    cmds = [f"read_verilog -sv {comb}", "prep -top fwd -flatten",
            "memory_map", "techmap", "opt -fast"]
    for (x, xd, th, thd) in states:
        cmds.append(
            f"eval -set X {vlit(x)} -set XD {vlit(xd)} "
            f"-set TH {vlit(th)} -set THD {vlit(thd)} -show z"
        )
    r = subprocess.run(["yosys", "-p", "; ".join(cmds)],
                       capture_output=True, text=True, errors="replace")
    # yosys imprime "Eval result: \z = 19082965." ou "... = 32'1111...0110."
    got = []
    for tok in re.findall(r"Eval result: \\z = ([^.\s]+)\.", r.stdout):
        m = re.fullmatch(r"(\d+)'([01]+)", tok)
        got.append(from_bits(m.group(2), len(m.group(2))) if m else int(tok))

    os.remove(comb)

    if len(got) != len(states):
        print("FALHA: yosys nao retornou um z por estado.")
        print(f"  esperados {len(states)}, obtidos {len(got)}")
        print(r.stdout[-1500:] or r.stderr[-1500:])
        return 2

    bad = 0
    print(f"{'x':>7} {'xd':>7} {'th':>5} {'thd':>7} "
          f"{'z (Python)':>12} {'z (Verilog)':>12}  ok")
    for (st, gz) in zip(states, got):
        rz = forward_ref(*st, W)
        ok = rz == gz
        bad += not ok
        print(f"{st[0]:>7} {st[1]:>7} {st[2]:>5} {st[3]:>7} "
              f"{rz:>12} {gz:>12}  {'sim' if ok else 'NAO'}")
    print()
    if bad:
        print(f"DIVERGENCIA em {bad}/{len(states)} estados — "
              f"o Verilog NAO reproduz a aritmetica verificada.")
        return 1
    print(f"OK — {len(states)}/{len(states)} estados batem exatamente. "
          f"O modelo IC3 usa o mesmo controlador que o harness ESBMC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
