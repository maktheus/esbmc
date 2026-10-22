#!/usr/bin/env python3
"""Gera mutantes SABIDAMENTE INSEGUROS do sistema real. Um pipeline que so sabe 'provar'
nao e evidencia: cada mutante deve resultar em CONTRAEXEMPLO (exit 1 do run_pdr.sh)."""
import pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
base = (HERE / "real" / "cl_ddpg16.v").read_text()
M = {
 "m1_init_thd_livre": ("thd <= ci(i_thd);", "thd <= i_thd;"),            # velocidade inicial fora da regiao
 "m2_atuador_invertido": ("wire signed [31:0] F_Q = tanh_z * 32'sd10;", "wire signed [31:0] F_Q = -(tanh_z * 32'sd10);"),
 "m3_controle_desligado": ("wire signed [31:0] F_Q = tanh_z * 32'sd10;", "wire signed [31:0] F_Q = 32'sd0;"),
  "m5_init_th_livre": ("th <= ci(i_th);", "th <= i_th;"),   # profundidade 0
 "m4_limite_apertado": ("(th > 16'sd53) || (th < -16'sd53)", "(th > 16'sd6) || (th < -16'sd6)"),
}
for name, (a, b) in M.items():
    assert a in base, f"padrao nao encontrado para {name}: {a}"
    (HERE / "mut").mkdir(exist_ok=True)
    (HERE / "mut" / f"{name}.v").write_text(base.replace(a, b))
    print("gerado", name)
