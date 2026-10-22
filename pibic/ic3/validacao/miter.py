#!/usr/bin/env python3
"""Miter SAT: transicao gerada (td por deslocamento) vs referencia independente com `/` (semantica C).
Entradas: 4 estados de 16 bits (sign-extended). Saidas comparadas: n_x,n_xd,n_th,n_thd e bad."""
import sys, pathlib, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gen_transition_system import WEIGHTS, load_weights, emit, TH_BND, vconst

def ref_module(W):
    H1, H2 = len(W["b1"]), len(W["b2"]); L = []
    L.append("module refm(input signed [15:0] x, xd, th, thd, output signed [31:0] nx, nxd, nth, nthd, output bad);")
    L.append("  wire signed [31:0] X = x, XD = xd, TH = th, THD = thd;")
    for i in range(H1):
        a, b, c, d = W["w1"][i]
        L.append(f"  wire signed [31:0] p1_{i} = ({vconst(a)}*X)/32'sd256 + ({vconst(b)}*XD)/32'sd256 + ({vconst(c)}*TH)/32'sd256 + ({vconst(d)}*THD)/32'sd256 + {vconst(W['b1'][i])};")
        L.append(f"  wire signed [31:0] a1_{i} = (p1_{i} > 0) ? p1_{i} : 32'sd0;")
    for j in range(H2):
        t = " + ".join(f"({vconst(W['w2'][j][k])}*a1_{k})/32'sd256" for k in range(H1))
        L.append(f"  wire signed [31:0] p2_{j} = {t} + {vconst(W['b2'][j])};")
        L.append(f"  wire signed [31:0] a2_{j} = (p2_{j} > 0) ? p2_{j} : 32'sd0;")
    o = " + ".join(f"({vconst(W['w_out'][0][k])}*a2_{k})/32'sd256" for k in range(H2))
    L.append(f"  wire signed [31:0] zz = {o} + {vconst(W['b_out'][0])};")
    L.append("  wire signed [31:0] za = (zz < 0) ? -zz : zz;")
    L.append("  wire signed [31:0] ta = (za <= 64) ? (za*252)/256 : (za <= 192) ? 62 + ((za-64)*200)/256 : (za <= 384) ? 162 + ((za-192)*92)/256 : (za <= 768) ? 231 + ((za-384)*16)/256 : 255;")
    L.append("  wire signed [31:0] tz = (zz < 0) ? -ta : ta;")
    L.append("  wire signed [31:0] FQ = tz*10;")
    L.append("  wire signed [31:0] tacc = (4040*TH - 375*FQ)/256;")
    L.append("  assign nx = X + (5*XD)/256; assign nxd = XD + (5*FQ)/256; assign nth = TH + (5*THD)/256; assign nthd = THD + (5*tacc)/256;")
    L.append(f"  assign bad = (th > {TH_BND}) || (th < -{TH_BND});")
    L.append("endmodule")
    return "\n".join(L)

def gen_module(bits=16):
    src = emit(bits)
    # transforma o 'top' gerado em modulo combinacional exposto: estado vira entrada, n_* vira saida
    src = src.replace("module top(input clk,", "module gen(input clk,")
    src = src.replace(f"input signed [{bits-1}:0] i_x, i_xd, i_th, i_thd,", f"input signed [{bits-1}:0] i_x, i_xd, i_th, i_thd,")
    return src

if __name__ == "__main__":
    W = load_weights(WEIGHTS)
    top = emit(16)
    # expoe o proximo estado do gerador sem reescrever sua logica: adiciona portas de saida
    top = top.replace("output bad);", "output bad, output signed [31:0] o_nx, o_nxd, o_nth, o_nthd);", 1)
    top = top.replace("endmodule", "  assign o_nx = n_x; assign o_nxd = n_xd; assign o_nth = n_th; assign o_nthd = n_thd;\nendmodule")
    pathlib.Path("miter_gen.v").write_text(top)
    pathlib.Path("miter_ref.v").write_text(ref_module(W) + "\n")
    print("gerados miter_gen.v e miter_ref.v")
