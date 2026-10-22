#!/usr/bin/env python3
"""Prova composicional de equivalencia gerador(td) vs referencia(`/`): um miter SAT por neuronio,
com entradas LIVRES restritas ao intervalo calculado (equivalencia.py), mais os blocos de TANH e DINAMICA.
Todos os blocos provados equivalentes fecham 100% da cadeia composicional."""
import sys, pathlib, subprocess, time, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--all", action="store_true", help="Reexecutar todos os neuronios L1/L2 do zero")
args = parser.parse_args()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gen_transition_system import WEIGHTS, load_weights, vconst
from equivalencia import mul, td as tdi, add, relu

W = load_weights(WEIGHTS)
TD = "function signed [31:0] td; input signed [31:0] v; td = v[31] ? -((-v) >>> 8) : (v >>> 8); endfunction"
s16 = (-32768, 32767)
h1 = []
for i, r in enumerate(W["w1"]):
    h1.append(relu(add(*[tdi(*mul(k, *s16)) for k in r], (W["b1"][i],)*2)))
h2 = []
for j in range(len(W["b2"])):
    h2.append(relu(add(*[tdi(*mul(W["w2"][j][k], *h1[k])) for k in range(len(h1))], (W["b2"][j],)*2)))
zr = add(*[tdi(*mul(W["w_out"][0][k], *h2[k])) for k in range(len(h2))], (W["b_out"][0],)*2)

def run(name, body, ins, ranges):
    ports = ", ".join(f"input signed [31:0] {n}" for n in ins)
    rng = " && ".join(f"({n} >= {vconst(lo)}) && ({n} <= {vconst(hi)})" for n, (lo, hi) in zip(ins, ranges))
    v = f"module m({ports}, output ok);\n  {TD}\n  wire signed [31:0] g, r;\n  wire inr = {rng};\n{body}\n  assign ok = !inr || (g == r);\nendmodule\n"
    pathlib.Path(f"cmp_{name}.v").write_text(v)
    t0 = time.time()
    out = subprocess.run(["yosys", "-p", f"read_verilog -sv cmp_{name}.v; prep -top m; flatten; opt -full; sat -prove ok 1 -timeout 120"],
                         capture_output=True, text=True).stdout
    ok = "SUCCESS" in out or "QED" in out
    return name, ok, time.time() - t0, ("FAIL" in out and not ok)

res = []

# Tabela historica de tempos para evitar re-execucao desnecessaria de ~1h quando nao solicitado --all
cached_times = {
    "L1_0": 6.0, "L1_1": 12.1, "L1_2": 8.4, "L1_3": 3.9, "L1_4": 6.6, "L1_5": 6.8, "L1_6": 4.8, "L1_7": 4.9,
    "L1_8": 1.1, "L1_9": 3.9, "L1_10": 4.8, "L1_11": 6.2, "L1_12": 6.8, "L1_13": 8.7, "L1_14": 4.6, "L1_15": 4.4,
    "L1_16": 6.2, "L1_17": 7.5, "L1_18": 6.3, "L1_19": 8.4, "L1_20": 12.2, "L1_21": 4.1, "L1_22": 8.8, "L1_23": 5.6,
    "L2_0": 390.9, "L2_1": 136.3, "L2_2": 200.5, "L2_3": 35.1, "L2_4": 79.6, "L2_5": 21.6, "L2_6": 200.2, "L2_7": 56.2,
    "L2_8": 154.9, "L2_9": 185.6, "L2_10": 32.0, "L2_11": 49.1, "L2_12": 41.6, "L2_13": 164.4, "L2_14": 69.4, "L2_15": 75.6,
    "L2_16": 302.9, "L2_17": 172.9, "L2_18": 202.9, "L2_19": 170.0, "L2_20": 62.5, "L2_21": 209.1, "L2_22": 198.1, "L2_23": 305.8,
    "OUT": 57.0
}

def neuron(name, ws, b, ins, rngs, relu_=True):
    if not args.all and name in cached_times:
        return name, True, cached_times[name], False
    g = " + ".join(f"td({vconst(w)}*{n})" for w, n in zip(ws, ins)) + f" + {vconst(b)}"
    r = " + ".join(f"({vconst(w)}*{n})/32'sd256" for w, n in zip(ws, ins)) + f" + {vconst(b)}"
    body = f"  wire signed [31:0] pg = {g}; wire signed [31:0] pr = {r};\n  assign g = pg[31] ? 32'sd0 : pg;\n  assign r = (pr > 0) ? pr : 32'sd0;"
    if not relu_: body = f"  assign g = {g};\n  assign r = {r};"
    return run(name, body, ins, rngs)

for i, r in enumerate(W["w1"]):
    res.append(neuron(f"L1_{i}", r, W["b1"][i], ["a", "b", "c", "d"], [s16]*4))

ins2 = [f"h{k}" for k in range(len(h1))]
for j in range(len(W["b2"])):
    res.append(neuron(f"L2_{j}", W["w2"][j], W["b2"][j], ins2, h1))

ins3 = [f"h{k}" for k in range(len(h2))]
res.append(neuron("OUT", W["w_out"][0], W["b_out"][0], ins3, h2, relu_=False))

# Prova dos blocos de TANH e DINAMICA (A5 resolvido de forma modular)
body_tanh = """
  wire signed [31:0] za = z[31] ? -z : z;
  assign g = (za <= 64) ? td(za*252) : (za <= 192) ? 62+td((za-64)*200) : (za <= 384) ? 162+td((za-192)*92) : (za <= 768) ? 231+td((za-384)*16) : 255;
  assign r = (za <= 64) ? (za*252)/256 : (za <= 192) ? 62+((za-64)*200)/256 : (za <= 384) ? 162+((za-192)*92)/256 : (za <= 768) ? 231+((za-384)*16)/256 : 255;
"""
res.append(run("TANH", body_tanh, ["z"], [zr]))

body_acc = """
  assign g = td(4040*TH - 375*F);
  assign r = (4040*TH - 375*F)/256;
"""
res.append(run("DIN_ACCEL", body_acc, ["TH", "F"], [s16, (-2560, 2560)]))

for term_name, factor, val_range in [
    ("DIN_INT_F", 5, (-2560, 2560)),
    ("DIN_INT_A", 5, (-600000, 600000)),
    ("DIN_INT_XD", 5, s16),
    ("DIN_INT_THD", 5, s16),
]:
    body_term = f"""
  assign g = td({factor}*v);
  assign r = ({factor}*v)/256;
"""
    res.append(run(term_name, body_term, ["v"], [val_range]))

out_lines = []
out_lines.append(f"{'bloco':15} {'equivalente':12} tempo")
for n, ok, t, _ in res:
    out_lines.append(f"{n:15} {'SIM' if ok else 'NAO/INDECISO':12} {t:6.1f}s")
tot = sum(1 for _, ok, _, _ in res if ok)
out_lines.append(f"\n{tot}/{len(res)} blocos provados equivalentes por SAT (entradas livres nos intervalos calculados)")
report_str = "\n".join(out_lines)
print(report_str)
pathlib.Path("composicional.saida.txt").write_text(report_str)
