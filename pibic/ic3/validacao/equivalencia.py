#!/usr/bin/env python3
"""Equivalencia formal do forward+dinamica gerados vs semantica C (`/` trunca p/ zero).
1) interval analysis sobre TODO o caminho, estado de B bits -> mostra se algum argumento de td
   pode chegar a INT_MIN ou se alguma soma estoura 32 bits;
2) miter SAT (yosys) entre o gerador (td) e uma referencia independente escrita com `/`."""
import json, sys, subprocess, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gen_transition_system import WEIGHTS, load_weights

I32 = (-2**31, 2**31 - 1)
def chk(name, lo, hi, rep):
    ok = lo >= I32[0] and hi <= I32[1]
    intmin = lo <= I32[0]
    rep.append((name, lo, hi, ok, intmin))
def mul(c, lo, hi):
    a, b = c*lo, c*hi
    return (min(a, b), max(a, b))
def td(lo, hi):  # monotono, trunca p/ zero
    f = lambda v: -((-v) >> 8) if v < 0 else v >> 8
    return f(lo), f(hi)
def add(*r): return (sum(x[0] for x in r), sum(x[1] for x in r))
def relu(r): return (max(r[0], 0), max(r[1], 0))

def analyse(bits):
    W = load_weights(WEIGHTS); rep = []
    s = (-2**(bits-1), 2**(bits-1) - 1)
    h1 = []
    for i, (a, b, c, d) in enumerate(W["w1"]):
        prods = [mul(k, *s) for k in (a, b, c, d)]
        for p in prods: chk("arg td L1", *p, rep)
        pre = add(*[td(*p) for p in prods], (W["b1"][i],)*2); chk("pre1", *pre, rep); h1.append(relu(pre))
    h2 = []
    for j in range(len(W["b2"])):
        prods = [mul(W["w2"][j][k], *h1[k]) for k in range(len(h1))]
        for p in prods: chk("arg td L2", *p, rep)
        pre = add(*[td(*p) for p in prods], (W["b2"][j],)*2); chk("pre2", *pre, rep); h2.append(relu(pre))
    prods = [mul(W["w_out"][0][k], *h2[k]) for k in range(len(h2))]
    for p in prods: chk("arg td out", *p, rep)
    z = add(*[td(*p) for p in prods], (W["b_out"][0],)*2); chk("z", *z, rep)
    zab = (0, max(abs(z[0]), abs(z[1]))); chk("-z", -z[1], -z[0], rep)
    # cada ramo do tanh so executa dentro da sua guarda (z_abs <= 64, <=192, <=384, <=768)
    for name, k, lo, hi in (("z_abs*252", 252, 0, 64), ("(z_abs-64)*200", 200, 1, 128),
                            ("(z_abs-192)*92", 92, 1, 192), ("(z_abs-384)*16", 16, 1, 384)):
        chk("arg td tanh "+name, *mul(k, lo, hi), rep)
    FQ = (-255*10, 255*10); chk("F_Q", *FQ, rep)
    TH = s
    chk("arg td 4040*TH-375*F_Q", *add(mul(4040, *TH), mul(-375, *FQ)), rep)
    tha = td(*add(mul(4040, *TH), mul(-375, *FQ))); chk("th_acc", *tha, rep)
    for nm, k, r in (("5*XD", 5, s), ("5*F_Q", 5, FQ), ("5*THD", 5, s), ("5*th_acc", 5, tha)):
        chk("arg td "+nm, *mul(k, *r), rep)
    chk("n_x", *add(s, td(*mul(5, *s))), rep); chk("n_xd", *add(s, td(*mul(5, *FQ))), rep)
    return rep

if __name__ == "__main__":
    for bits in (16, 32):
        rep = analyse(bits)
        worst = max(rep, key=lambda r: max(abs(r[1]), abs(r[2])))
        bad = [r for r in rep if not r[3]]
        im = [r for r in rep if r[4]]
        print(f"estado {bits} bits: {len(rep)} expressoes analisadas; maior |valor| = {max(abs(worst[1]),abs(worst[2])):,} ({worst[0]}), limite 2^31 = {2**31:,}")
        print(f"   estouram 32 bits: {len(bad)}; podem atingir INT_MIN: {len(im)}")
        for r in bad[:5]: print("     ", r[0], r[1], r[2])
