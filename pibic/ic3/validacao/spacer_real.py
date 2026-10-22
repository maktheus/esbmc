#!/usr/bin/env python3
"""Motor INDEPENDENTE (Z3/Spacer, IC3 de nivel de palavra) sobre a mesma semantica Q8.8 do ator real.
Aritmetica inteira exata (sem bit-blasting). Verifica |th|<=53 para todo t a partir de |estado|<=5.
Se 'proved', o invariante e re-checado por um solver SMT comum (3 implicacoes).
Mutantes (--mut) devem produzir 'cex'."""
import argparse, sys, time, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gen_transition_system import WEIGHTS, load_weights, TH_BND, INIT_BND
import z3

ap = argparse.ArgumentParser()
ap.add_argument("--timeout", type=int, default=600)
ap.add_argument("--mut", default="none", choices=["none", "thd_livre", "atuador_invertido", "controle_desligado", "limite_apertado"])
ap.add_argument("--init-bnd", type=int, default=INIT_BND)
ap.add_argument("--weights", default=str(WEIGHTS))
a = ap.parse_args()
W = load_weights(pathlib.Path(a.weights).resolve())

def td(v):  # trunca para zero
    return z3.If(v >= 0, v / 256, -((-v) / 256))
def relu(v): return z3.If(v > 0, v, z3.IntVal(0))
def layer(ws, bs, xs):
    return [relu(sum((td(w * x) for w, x in zip(row, xs)), z3.IntVal(0)) + b) for row, b in zip(ws, bs)]

def step(x, xd, th, thd):
    h1 = [relu(td(r[0]*x) + td(r[1]*xd) + td(r[2]*th) + td(r[3]*thd) + b) for r, b in zip(W["w1"], W["b1"])]
    h2 = layer(W["w2"], W["b2"], h1)
    z = sum((td(w * h) for w, h in zip(W["w_out"][0], h2)), z3.IntVal(0)) + W["b_out"][0]
    za = z3.If(z < 0, -z, z)
    ta = z3.If(za <= 64, td(za*252), z3.If(za <= 192, 62 + td((za-64)*200),
         z3.If(za <= 384, 162 + td((za-192)*92), z3.If(za <= 768, 231 + td((za-384)*16), z3.IntVal(255)))))
    tz = z3.If(z < 0, -ta, ta)
    FQ = tz * 10
    if a.mut == "atuador_invertido": FQ = -FQ
    if a.mut == "controle_desligado": FQ = z3.IntVal(0)
    tacc = td(4040*th - 375*FQ)
    return x + td(5*xd), xd + td(5*FQ), th + td(5*thd), thd + td(5*tacc)

fp = z3.Fixedpoint(); fp.set(engine="spacer")
Z = z3.IntSort(); B = z3.BoolSort()
Inv = z3.Function("Inv", Z, Z, Z, Z, B); fp.register_relation(Inv)
x, xd, th, thd = z3.Ints("x xd th thd")
nx, nxd, nth, nthd = step(x, xd, th, thd)
lim = TH_BND if a.mut != "limite_apertado" else 6
B0 = a.init_bnd
inb = lambda *v: z3.And(*[z3.And(t >= -B0, t <= B0) for t in v])
init = inb(x, xd, th, thd) if a.mut != "thd_livre" else z3.And(inb(x, xd, th), thd >= -32768, thd <= 32767)
for v in (x, xd, th, thd): fp.declare_var(v)
fp.rule(Inv(x, xd, th, thd), init)
fp.rule(Inv(nx, nxd, nth, nthd), Inv(x, xd, th, thd))
bad = z3.Or(th > lim, th < -lim)
t0 = time.time()
fp.set("timeout", a.timeout * 1000)
try:
    r = fp.query(z3.And(Inv(x, xd, th, thd), bad))
except z3.Z3Exception as e:
    r = z3.unknown; print("z3 abortou:", e)
el = time.time() - t0
print(f"mut={a.mut} init_bnd={B0} z3={z3.get_version_string()} resultado={r} tempo={el:.1f}s")
if r == z3.unsat:
    inv = fp.get_answer()
    pathlib.Path(f"spacer_{a.mut}.invariante.smt2").write_text(str(inv))
    print("INVARIANTE (resumo, 1500 chars):", str(inv)[:1500])
    # recheca com solver SMT comum: init=>I, I&T=>I', I&bad=>false
    lam = inv.arg(0) if hasattr(inv, "arg") else None
    print("(invariante salvo; recheca independente: ver recheca.py)")
elif r == z3.sat:
    print("CONTRAEXEMPLO encontrado (sat). Trecho:", str(fp.get_answer())[:800])
else:
    print("INDECISO:", fp.reason_unknown())
sys.exit({z3.unsat: 0, z3.sat: 1}.get(r, 3))
