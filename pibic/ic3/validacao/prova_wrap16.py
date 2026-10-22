#!/usr/bin/env python3
"""Prova formal da ausencia de wrap-around de 16 bits na transicao do cart-pole.
Verifica que para qualquer estado dentro do envelope operacional e da regiao de seguranca,
as proximas variaveis de estado n_x, n_xd, n_th, n_thd NUNCA sofrem overflow/wrap em 16 bits
(ou seja, n_var == $signed(n_var[15:0]) para todas as variaveis)."""

import subprocess, time, pathlib

TD = "function signed [31:0] td; input signed [31:0] v; td = v[31] ? -((-v) >>> 8) : (v >>> 8); endfunction"

# Verilog module checando ausencia de wrap para todo estado no dominio de operacao
v_code = f"""module m(
  input signed [31:0] X,
  input signed [31:0] XD,
  input signed [31:0] TH,
  input signed [31:0] THD,
  input signed [31:0] F_Q,
  output ok
);
  {TD}

  // Dominio de operacao amplo (ate 10x a regiao de seguranca)
  // X in [-6140, 6140] (ate 24m)
  // XD in [-12800, 12800] (ate 50m/s)
  // TH in [-530, 530] (ate 120 graus)
  // THD in [-12800, 12800] (ate 50 rad/s)
  // F_Q in [-2560, 2560] (limitado pela saturacao do tanh)
  wire in_domain = (X >= -32'sd6140) && (X <= 32'sd6140) &&
                   (XD >= -32'sd12800) && (XD <= 32'sd12800) &&
                   (TH >= -32'sd530) && (TH <= 32'sd530) &&
                   (THD >= -32'sd12800) && (THD <= 32'sd12800) &&
                   (F_Q >= -32'sd2560) && (F_Q <= 32'sd2560);

  wire signed [31:0] th_acc = td(32'sd4040*TH - 32'sd375*F_Q);
  wire signed [31:0] n_x   = X   + td(32'sd5*XD);
  wire signed [31:0] n_xd  = XD  + td(32'sd5*F_Q);
  wire signed [31:0] n_th  = TH  + td(32'sd5*THD);
  wire signed [31:0] n_thd = THD + td(32'sd5*th_acc);

  // Verificacao: n_var cabe em signed 16 bits [-32768, 32767]
  wire no_wrap_x   = (n_x   >= -32'sd32768) && (n_x   <= 32'sd32767);
  wire no_wrap_xd  = (n_xd  >= -32'sd32768) && (n_xd  <= 32'sd32767);
  wire no_wrap_th  = (n_th  >= -32'sd32768) && (n_th  <= 32'sd32767);
  wire no_wrap_thd = (n_thd >= -32'sd32768) && (n_thd <= 32'sd32767);

  wire all_no_wrap = no_wrap_x && no_wrap_xd && no_wrap_th && no_wrap_thd;

  assign ok = !in_domain || all_no_wrap;
endmodule
"""

pathlib.Path("cmp_nowrap16.v").write_text(v_code)
t0 = time.time()
res = subprocess.run([
    "yosys", "-p",
    "read_verilog -sv cmp_nowrap16.v; prep -top m; flatten; opt -full; sat -prove ok 1 -timeout 60"
], capture_output=True, text=True)
dt = time.time() - t0
out = res.stdout

if "SUCCESS" in out or "QED" in out:
    print(f"PROVA DE AUSENCIA DE WRAP 16 BITS: SUCESSO (tempo: {dt:.2f}s)")
    print("Nenhuma variavel de estado (X, XD, TH, THD) sofre overflow de 16 bits na dinamica.")
else:
    print("FALHA OU INDECISO:", out[-300:])
