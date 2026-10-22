#!/usr/bin/env python3
import subprocess, time, sys

TD = "function signed [31:0] td; input signed [31:0] v; td = v[31] ? -((-v) >>> 8) : (v >>> 8); endfunction"

tests = [
    ("TANH_BRANCH_0", """
  wire inr = (za >= 0) && (za <= 64);
  wire signed [31:0] tg = td(za * 252);
  wire signed [31:0] tr = (za * 252) / 256;
  assign ok = !inr || (tg == tr);
""", ["za"], [(0, 64)]),

    ("TANH_BRANCH_1", """
  wire inr = (za > 64) && (za <= 192);
  wire signed [31:0] tg = 62 + td((za - 64) * 200);
  wire signed [31:0] tr = 62 + ((za - 64) * 200) / 256;
  assign ok = !inr || (tg == tr);
""", ["za"], [(65, 192)]),

    ("TANH_BRANCH_2", """
  wire inr = (za > 192) && (za <= 384);
  wire signed [31:0] tg = 162 + td((za - 192) * 92);
  wire signed [31:0] tr = 162 + ((za - 192) * 92) / 256;
  assign ok = !inr || (tg == tr);
""", ["za"], [(193, 384)]),

    ("TANH_BRANCH_3", """
  wire inr = (za > 384) && (za <= 768);
  wire signed [31:0] tg = 231 + td((za - 384) * 16);
  wire signed [31:0] tr = 231 + ((za - 384) * 16) / 256;
  assign ok = !inr || (tg == tr);
""", ["za"], [(385, 768)]),

    ("TANH_BRANCH_4", """
  wire inr = (za > 768) && (za <= 32'sd30000000);
  wire signed [31:0] tg = 255;
  wire signed [31:0] tr = 255;
  assign ok = !inr || (tg == tr);
""", ["za"], [(769, 30000000)]),

    ("TANH_FULL", """
  wire inr = (z >= (-32'sd20037774)) && (z <= 32'sd22845636);
  wire signed [31:0] za = z[31] ? -z : z;
  wire signed [31:0] tg = (za <= 64) ? td(za*252) : (za <= 192) ? 62+td((za-64)*200) : (za <= 384) ? 162+td((za-192)*92) : (za <= 768) ? 231+td((za-384)*16) : 255;
  wire signed [31:0] tr = (za <= 64) ? (za*252)/256 : (za <= 192) ? 62+((za-64)*200)/256 : (za <= 384) ? 162+((za-192)*92)/256 : (za <= 768) ? 231+((za-384)*16)/256 : 255;
  assign ok = !inr || (tg == tr);
""", ["z"], [(-20037774, 22845636)]),

    ("DIN_ACCEL", """
  wire inr = (TH >= (-32'sd32768)) && (TH <= 32'sd32767) && (F >= (-32'sd2560)) && (F <= 32'sd2560);
  wire signed [31:0] ag = td(4040*TH - 375*F);
  wire signed [31:0] ar = (4040*TH - 375*F)/256;
  assign ok = !inr || (ag == ar);
""", ["TH", "F"], [(-32768, 32767), (-2560, 2560)]),

    ("DIN_TERM_5F", """
  wire inr = (F >= (-32'sd2560)) && (F <= 32'sd2560);
  wire signed [31:0] g = td(5*F);
  wire signed [31:0] r = (5*F)/256;
  assign ok = !inr || (g == r);
""", ["F"], [(-2560, 2560)]),

    ("DIN_TERM_5A", """
  wire inr = (A >= (-32'sd600000)) && (A <= 32'sd600000);
  wire signed [31:0] g = td(5*A);
  wire signed [31:0] r = (5*A)/256;
  assign ok = !inr || (g == r);
""", ["A"], [(-600000, 600000)]),

    ("DIN_TERM_5XD", """
  wire inr = (XD >= (-32'sd32768)) && (XD <= 32'sd32767);
  wire signed [31:0] g = td(5*XD);
  wire signed [31:0] r = (5*XD)/256;
  assign ok = !inr || (g == r);
""", ["XD"], [(-32768, 32767)]),

    ("DIN_TERM_5THD", """
  wire inr = (THD >= (-32'sd32768)) && (THD <= 32'sd32767);
  wire signed [31:0] g = td(5*THD);
  wire signed [31:0] r = (5*THD)/256;
  assign ok = !inr || (g == r);
""", ["THD"], [(-32768, 32767)])
]

print(f"{'Caso':20} {'Status':12} {'Tempo':8}")
print("-" * 42)

for name, body, ins, ranges in tests:
    ports = ", ".join(f"input signed [31:0] {n}" for n in ins)
    v = f"module m({ports}, output ok);\n  {TD}\n{body}\nendmodule\n"
    fname = f"cmp_{name}.v"
    with open(fname, "w") as f:
        f.write(v)
    t0 = time.time()
    res = subprocess.run([
        "yosys", "-p",
        f"read_verilog -sv {fname}; prep -top m; flatten; opt -full; sat -prove ok 1 -timeout 60"
    ], capture_output=True, text=True)
    dt = time.time() - t0
    out = res.stdout
    if "SUCCESS" in out or "QED" in out:
        status = "PROVADO"
    elif "FAIL" in out:
        status = "FALHOU"
    else:
        status = "TIMEOUT"
    print(f"{name:20} {status:12} {dt:6.2f}s")
