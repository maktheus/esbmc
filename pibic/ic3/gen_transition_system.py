#!/usr/bin/env python3
"""
gen_transition_system.py — Emite o sistema de transição da malha fechada
cart-pole + Actor DDPG quantizado como Verilog, para model checking indutivo
(IC3/PDR) via yosys + ABC.

MOTIVAÇÃO
---------
A verificação em malha fechada com BMC (ESBMC) é de 1 passo. Medição nesta
máquina, com uma rede de 24 neurônios:

    K=1    0,3 s     46 MB
    K=2     77 s    137 MB
    K=4    909 s    383 MB

O custo explode porque o BMC constrói UMA fórmula com K cópias da relação de
transição. O harness de 50 passos (`cartpole/closedloop_esbmc_stub.c`) nunca foi
implementado por essa razão — é limite do método, não de esforço.

IC3/PDR nunca constrói essa fórmula: cada consulta é sobre UMA cópia da
transição, logo o pico de memória é independente da profundidade da prova. E o
resultado é qualitativamente diferente — um invariante indutivo, isto é
"seguro para sempre", em vez de "seguro até K".

FIDELIDADE À ARITMÉTICA VERIFICADA
----------------------------------
O forward foi implementado para replicar `generate_controller_body` de
`cartpole/verify_ddpg_closed_loop.py:35-65`:

  - divisão por 256 **por termo**, não soma-depois-divide;
  - truncamento para zero (o `/` de inteiro do C), não deslocamento aritmético,
    que arredonda para -infinito e daria outro resultado para negativos;
  - ReLU como `pre > 0 ? pre : 0`;
  - a mesma aproximação linear de tanh em 5 ramos (`TANH_APPROX_C`);
  - a mesma dinâmica linearizada, coeficientes 4040 e 375.

UMA DIFERENÇA DELIBERADA: o harness C injeta
`__ESBMC_assume(pre1_i >= lo && pre1_i <= hi)` com limites de propagação de
intervalo. Isso é muleta do BMC — poda o espaço para a fórmula caber. Aqui
esses assumes são OMITIDOS, então o modelo é estritamente mais fiel: nenhum
estado alcançável é descartado. Ver KB-C04.

`validate_forward.py` faz teste diferencial em estados concretos. Ele detecta
divergências, mas a amostragem, sozinha, não prova equivalência universal.

Uso:
    python3 gen_transition_system.py --bits 32 -o cl_ddpg32.v
    python3 gen_transition_system.py --bits 16 -o cl_ddpg16.v
"""

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WEIGHTS = (
    HERE / ".." / "cartpole" / "webapp" / "public" / "ddpg_weights_q88.json"
).resolve()

SCALE = 256
TH_BND = int(0.2094 * SCALE)   # 53 -> 12 graus, o limite de "pendulo em pe"
INIT_BND = 5                   # regiao inicial: |estado| <= 5 (perto do equilibrio)


def vconst(v: int) -> str:
    """Constante assinada de 32 bits válida em Verilog (32'sd-N é ilegal)."""
    return f"32'sd{v}" if v >= 0 else f"(-32'sd{-v})"


def load_weights(path: Path) -> dict[str, Any]:
    """Carrega o ator e rejeita formatos que gerariam Verilog incorreto."""
    with path.open(encoding="utf-8") as fh:
        w = json.load(fh)

    required = {"w1", "b1", "w2", "b2", "w_out", "b_out"}
    missing = sorted(required - w.keys()) if isinstance(w, dict) else sorted(required)
    if missing:
        raise ValueError(f"pesos sem os campos obrigatorios: {', '.join(missing)}")

    w1, b1, w2, b2 = w["w1"], w["b1"], w["w2"], w["b2"]
    w_out, b_out = w["w_out"], w["b_out"]
    matrices = (w1, w2, w_out)
    vectors = (b1, b2, b_out)
    if (not all(isinstance(value, list) for value in matrices + vectors)
            or not all(all(isinstance(row, list) for row in matrix)
                       for matrix in matrices)):
        raise ValueError("matrizes e vetores de pesos devem ser listas JSON")
    if not b1 or not b2:
        raise ValueError("as duas camadas ocultas precisam ser nao vazias")
    if len(w1) != len(b1) or any(len(row) != 4 for row in w1):
        raise ValueError("w1 deve ter uma linha de 4 pesos para cada item de b1")
    if len(w2) != len(b2) or any(len(row) != len(b1) for row in w2):
        raise ValueError("w2 deve ter uma linha de len(b1) pesos para cada item de b2")
    if len(w_out) != 1 or len(w_out[0]) != len(b2) or len(b_out) != 1:
        raise ValueError("a camada de saida deve ter formato 1 x len(b2), com um bias")

    def all_ints(value: Any) -> bool:
        if isinstance(value, list):
            return all(all_ints(item) for item in value)
        return isinstance(value, int) and not isinstance(value, bool)

    if not all(all_ints(w[key]) for key in required):
        raise ValueError("todos os pesos e biases devem ser inteiros Q8.8")
    return w


def emit(bits: int, weights: Path = WEIGHTS) -> str:
    w = load_weights(weights)
    w1, b1 = w["w1"], w["b1"]
    w2, b2 = w["w2"], w["b2"]
    w_out, b_out = w["w_out"][0], w["b_out"][0]
    H1, H2 = len(b1), len(b2)

    L = []
    A = L.append
    A(f"// Gerado por gen_transition_system.py — NAO EDITAR A MAO")
    A(f"// Actor DDPG real: 4 -> {H1} -> {H2} -> 1, Q8.8, "
      f"{H1*4 + H1 + H2*H1 + H2 + H2 + 1} parametros")
    A(f"// Estado: {bits} bits por variavel. Propriedade: |th| <= {TH_BND} (12 graus)")
    A("")
    A("module top(input clk,")
    A(f"           input signed [{bits-1}:0] i_x, i_xd, i_th, i_thd,")
    A("           output bad);")
    A("")
    A("  reg started = 0;")
    A(f"  reg signed [{bits-1}:0] x = 0, xd = 0, th = 0, thd = 0;")
    A("")
    # estado promovido a 32 bits para a aritmetica (o C usa int de 32 bits)
    ext = lambda n: (f"{{{{{32-bits}{{{n}[{bits-1}]}}}},{n}}}" if bits < 32 else n)
    A(f"  wire signed [31:0] X   = {ext('x')};")
    A(f"  wire signed [31:0] XD  = {ext('xd')};")
    A(f"  wire signed [31:0] TH  = {ext('th')};")
    A(f"  wire signed [31:0] THD = {ext('thd')};")
    A("")
    A("  // divisao inteira por 256 truncando para ZERO — identica ao / do C.")
    A("  // NAO usar >>> 8 sozinho: arredonda para -infinito e divergiria")
    A("  // do harness verificado para valores negativos.")
    A("  function signed [31:0] td; input signed [31:0] v;")
    A("    td = v[31] ? -((-v) >>> 8) : (v >>> 8); endfunction")
    A("")
    A(f"  // clamp em [-{INIT_BND},{INIT_BND}] — sobrejetor sobre a regiao inicial exata")
    A(f"  function signed [{bits-1}:0] ci; input signed [{bits-1}:0] v;")
    A(f"    ci = (v >  {bits}'sd{INIT_BND}) ?  {bits}'sd{INIT_BND} :")
    A(f"         (v < -{bits}'sd{INIT_BND}) ? -{bits}'sd{INIT_BND} : v; endfunction")
    A("")

    # ── camada 1 ────────────────────────────────────────────────────────────
    A("  // camada 1: divisao POR TERMO, como verify_ddpg_closed_loop.py:43-44")
    for i in range(H1):
        a, b, c, d = w1[i]
        A(f"  wire signed [31:0] pre1_{i} = td({vconst(a)}*X) + td({vconst(b)}*XD)"
          f" + td({vconst(c)}*TH) + td({vconst(d)}*THD) + {vconst(b1[i])};")
        A(f"  wire signed [31:0] h1_{i} = pre1_{i}[31] ? 32'sd0 : pre1_{i};")
    A("")

    # ── camada 2 ────────────────────────────────────────────────────────────
    A("  // camada 2")
    for j in range(H2):
        terms = " + ".join(f"td({vconst(w2[j][k])}*h1_{k})" for k in range(H1))
        A(f"  wire signed [31:0] pre2_{j} = {terms} + {vconst(b2[j])};")
        A(f"  wire signed [31:0] h2_{j} = pre2_{j}[31] ? 32'sd0 : pre2_{j};")
    A("")

    # ── saida + tanh ────────────────────────────────────────────────────────
    A("  // saida")
    outt = " + ".join(f"td({vconst(w_out[k])}*h2_{k})" for k in range(H2))
    A(f"  wire signed [31:0] z = {outt} + {vconst(b_out)};")
    A("")
    A("  // aproximacao linear de tanh em Q8.8 — mesma de TANH_APPROX_C")
    A("  wire signed [31:0] z_abs = z[31] ? -z : z;")
    A("  wire signed [31:0] tanh_abs =")
    A("      (z_abs <=  32'sd64) ? td(z_abs * 32'sd252) :")
    A("      (z_abs <= 32'sd192) ?  32'sd62 + td((z_abs -  32'sd64) * 32'sd200) :")
    A("      (z_abs <= 32'sd384) ? 32'sd162 + td((z_abs - 32'sd192) *  32'sd92) :")
    A("      (z_abs <= 32'sd768) ? 32'sd231 + td((z_abs - 32'sd384) *  32'sd16) :")
    A("                            32'sd255;")
    A("  wire signed [31:0] tanh_z = z[31] ? -tanh_abs : tanh_abs;")
    A("  wire signed [31:0] F_Q = tanh_z * 32'sd10;   // (t*10*256)/256 == t*10")
    A("")
    A("  // dinamica linearizada Q8.8 — mesma de verify_ddpg_closed_loop.py:185-187")
    A("  wire signed [31:0] th_acc = td(32'sd4040*TH - 32'sd375*F_Q);")
    A("  wire signed [31:0] n_x   = X   + td(32'sd5*XD);")
    A("  wire signed [31:0] n_xd  = XD  + td(32'sd5*F_Q);")
    A("  wire signed [31:0] n_th  = TH  + td(32'sd5*THD);")
    A("  wire signed [31:0] n_thd = THD + td(32'sd5*th_acc);")
    A("")
    A("  always @(posedge clk) begin")
    A("    started <= 1;")
    A("    if (!started) begin")
    A("      x <= ci(i_x); xd <= ci(i_xd); th <= ci(i_th); thd <= ci(i_thd);")
    A("    end else begin")
    A(f"      x <= n_x[{bits-1}:0];   xd  <= n_xd[{bits-1}:0];")
    A(f"      th <= n_th[{bits-1}:0]; thd <= n_thd[{bits-1}:0];")
    A("    end")
    A("  end")
    A("")
    A(f"  // propriedade de seguranca: o pendulo nunca passa de 12 graus")
    A(f"  assign bad = (th > {bits}'sd{TH_BND}) || (th < -{bits}'sd{TH_BND});")
    A("endmodule")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=32, choices=[16, 32],
                   help="largura das variaveis de estado (32 = fiel ao int do C)")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--weights", type=Path, default=WEIGHTS,
                   help=f"JSON Q8.8 do ator (padrao: {WEIGHTS})")
    a = p.parse_args()
    try:
        src = emit(a.bits, a.weights.resolve())
        out = Path(a.out)
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(src)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    print(f"{out}: {len(src.splitlines())} linhas, estado {a.bits} bits "
          f"({4*a.bits + 1} flops)")


if __name__ == "__main__":
    main()
