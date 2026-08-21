#!/usr/bin/env python3
"""
validate_forward.py — Testa diferencialmente se o forward em Verilog gerado por
gen_transition_system.py produz o mesmo z que a aritmética Q8.8 de referência
(`cartpole/verify_ddpg_closed_loop.py:35-65`) nos estados selecionados.

Uma divergência demonstra que os modelos são diferentes. Concordância em uma
amostra aumenta a confiança e detecta regressões, mas não é prova universal de
equivalência para todos os valores de 32 bits.

Método: emite um módulo puramente combinacional (sem registradores) com os
mesmos pesos, sintetiza com yosys, e usa `yosys eval` para avaliar z em estados
concretos — comparando contra a referência Python termo a termo.

Uso:
    python3 validate_forward.py            # 12 estados
    python3 validate_forward.py -n 400     # amostragem maior
"""

import argparse
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_transition_system import WEIGHTS, load_weights, vconst, SCALE  # noqa: E402

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


def min_five(value: str) -> int:
    parsed = int(value)
    if parsed < 5:
        raise argparse.ArgumentTypeError("deve ser pelo menos 5 (cinco casos-limite fixos)")
    return parsed


def yosys_path(value: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        raise argparse.ArgumentTypeError(
            f"executavel '{value}' nao encontrado; instale Yosys ou use --yosys CAMINHO"
        )
    return resolved


def yosys_quote(path: Path) -> str:
    """Aspas aceitas pelo parser de comandos do Yosys, inclusive no Windows."""
    return '"' + path.resolve().as_posix().replace('"', '\\"') + '"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=min_five, default=12,
                    help="numero de estados a testar (minimo: 5)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--weights", type=Path, default=WEIGHTS,
                    help=f"JSON Q8.8 do ator (padrao: {WEIGHTS})")
    ap.add_argument("--yosys", type=yosys_path, default=None,
                    help="executavel do Yosys (ou defina YOSYS no ambiente)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="limite, em segundos, para a validacao completa")
    ap.add_argument("--keep-verilog", type=Path,
                    help="preserva o modulo combinacional neste caminho")
    a = ap.parse_args()
    if a.timeout <= 0:
        ap.error("--timeout deve ser positivo")

    yosys = a.yosys
    if yosys is None:
        configured = os.environ.get("YOSYS", "yosys")
        resolved = shutil.which(configured)
        if resolved is None:
            print("DEPENDENCIA AUSENTE: Yosys nao encontrado.", file=sys.stderr)
            print("Instale Yosys, defina YOSYS ou use --yosys CAMINHO.", file=sys.stderr)
            return 2
        yosys = resolved

    try:
        W = load_weights(a.weights.resolve())
    except (OSError, ValueError) as exc:
        print(f"FALHA ao carregar pesos: {exc}", file=sys.stderr)
        return 2

    temp_dir = None
    if a.keep_verilog:
        comb = a.keep_verilog.resolve()
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="ic3-fwd-")
        comb = Path(temp_dir.name) / "fwd_check.v"
    try:
        with comb.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(emit_comb(W))
    except OSError as exc:
        if temp_dir:
            temp_dir.cleanup()
        print(f"FALHA ao escrever Verilog temporario: {exc}", file=sys.stderr)
        return 2

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
    cmds = [f"read_verilog -sv {yosys_quote(comb)}", "prep -top fwd -flatten",
            "memory_map", "techmap", "opt -fast"]
    for (x, xd, th, thd) in states:
        cmds.append(
            f"eval -set X {vlit(x)} -set XD {vlit(xd)} "
            f"-set TH {vlit(th)} -set THD {vlit(thd)} -show z"
        )
    try:
        r = subprocess.run([yosys, "-p", "; ".join(cmds)],
                           capture_output=True, text=True, errors="replace",
                           timeout=a.timeout)
    except subprocess.TimeoutExpired:
        print(f"FALHA: Yosys excedeu o limite de {a.timeout:g} s.", file=sys.stderr)
        return_code = 2
        r = None
    finally:
        if temp_dir:
            temp_dir.cleanup()
    if r is None:
        return return_code
    if r.returncode != 0:
        print(f"FALHA: Yosys terminou com codigo {r.returncode}.", file=sys.stderr)
        print((r.stderr or r.stdout)[-1500:], file=sys.stderr)
        return 2
    # yosys imprime "Eval result: \z = 19082965." ou "... = 32'1111...0110."
    got = []
    for tok in re.findall(r"Eval result: \\z = ([^.\s]+)\.", r.stdout):
        m = re.fullmatch(r"(\d+)'([01]+)", tok)
        got.append(from_bits(m.group(2), len(m.group(2))) if m else int(tok))

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
    print(f"OK — {len(states)}/{len(states)} estados da amostra batem exatamente.")
    print("AVISO — teste diferencial por amostragem nao prova equivalencia universal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
