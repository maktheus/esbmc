set -e
python3 miter.py
# o gerador tem o estado em registradores; para comparar a funcao de transicao, congela o estado
# como entradas: remove regs e liga x,xd,th,thd as portas livres.
python3 - <<'PY'
s=open('miter_gen.v').read()
s=s.replace("reg started = 0;","")
s=s.replace("reg signed [15:0] x = 0, xd = 0, th = 0, thd = 0;","")
s=s.replace("input signed [15:0] i_x, i_xd, i_th, i_thd,","input signed [15:0] x, xd, th, thd,")
a=s.index("  always @(posedge clk)"); b=s.index("  // propriedade")
s=s[:a]+s[b:]
s=s.replace("module top(input clk,","module top(input clk,")
open('miter_gen_comb.v','w').write(s)
PY
cat > miter_top.v <<'V'
module miter(input signed [15:0] x, xd, th, thd, output ok);
  wire g_bad, r_bad; wire signed [31:0] gx, gxd, gth, gthd, rx, rxd, rth, rthd;
  top g(.clk(1'b0), .x(x), .xd(xd), .th(th), .thd(thd), .bad(g_bad), .o_nx(gx), .o_nxd(gxd), .o_nth(gth), .o_nthd(gthd));
  refm r(.x(x), .xd(xd), .th(th), .thd(thd), .nx(rx), .nxd(rxd), .nth(rth), .nthd(rthd), .bad(r_bad));
  assign ok = (gx==rx)&&(gxd==rxd)&&(gth==rth)&&(gthd==rthd)&&(g_bad==r_bad);
endmodule
V
timeout ${1:-3000} yosys -p "read_verilog -sv miter_gen_comb.v; read_verilog -sv miter_ref.v; read_verilog -sv miter_top.v; \
  prep -top miter; flatten; opt -full; sat -prove ok 1 -show x,xd,th,thd -timeout ${1:-3000}" 2>&1 | tail -25
