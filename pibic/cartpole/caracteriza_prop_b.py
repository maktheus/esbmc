#!/usr/bin/env python3
"""
caracteriza_prop_b.py — Encontra o maior |thd| para o qual o Actor DDPG real
garante segurança em dois passos.

MOTIVAÇÃO. A Property B original era vácua: asseria sobre `th_new`, que não
depende da saída do controlador (KB-C01). Reescrita para dois passos, ela passa
a depender de `F_Q` — mas falha a partir da caixa inicial declarada, com
`|thd| ≤ 1280` (5 rad/s). Isso não é defeito do controlador: com o pêndulo já
caindo a 5 rad/s, nenhuma força limitada a 10 N o segura em dois passos.

Em vez de afrouxar a propriedade até ela passar, este script mede **onde** ela
passa. O resultado — "o controlador garante segurança de dois passos para
|θ̇| ≤ X" — é uma caracterização do envelope de operação, e diz mais do que um
SUCCESSFUL obtido por construção.

Usa os pesos Q8.8 já quantizados (`ddpg_weights_q88.json`), não o `.pth`, para
não depender de torch.

Uso:
    python3 caracteriza_prop_b.py [--timeout 300]
"""

import argparse
import json
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
PIBIC = os.path.dirname(AQUI)
sys.path.insert(0, PIBIC)

from core_verify.esbmc_caller import SAFE, UNSAFE, run_esbmc  # noqa: E402

PESOS = os.path.join(AQUI, "webapp", "public", "ddpg_weights_q88.json")
SCALE, TH_BND, X_BND, XD_BND = 256, 53, 614, 1280

TANH = """
    int z_abs = z >= 0 ? z : -z;
    int tanh_abs;
    if (z_abs <= 64)        tanh_abs = (z_abs * 252) / 256;
    else if (z_abs <= 192)  tanh_abs = 62 + ((z_abs - 64) * 200) / 256;
    else if (z_abs <= 384)  tanh_abs = 162 + ((z_abs - 192) * 92) / 256;
    else if (z_abs <= 768)  tanh_abs = 231 + ((z_abs - 384) * 16) / 256;
    else                    tanh_abs = 255;
    int tanh_z = z >= 0 ? tanh_abs : -tanh_abs;
    int F_Q = tanh_z * 10;
"""


def corpo_controlador(w):
    """Mesma aritmética de generate_controller_body: divisão POR TERMO."""
    L = []
    H1, H2 = len(w["b1"]), len(w["b2"])
    for i in range(H1):
        a, b, c, d = w["w1"][i]
        L.append(f"    int pre1_{i} = (x*({a}))/256 + (xd*({b}))/256"
                 f" + (th*({c}))/256 + (thd*({d}))/256 + ({w['b1'][i]});")
        L.append(f"    int h1_{i} = pre1_{i} > 0 ? pre1_{i} : 0;")
    for j in range(H2):
        t = " + ".join(f"(h1_{k}*({w['w2'][j][k]}))/256" for k in range(H1))
        L.append(f"    int pre2_{j} = {t} + ({w['b2'][j]});")
        L.append(f"    int h2_{j} = pre2_{j} > 0 ? pre2_{j} : 0;")
    t = " + ".join(f"(h2_{k}*({w['w_out'][0][k]}))/256" for k in range(H2))
    L.append(f"    int z = {t} + ({w['b_out'][0]});")
    return "\n".join(L)


def harness(w, thd_bnd):
    return f"""\
/* Property B (2 passos) com |thd| <= {thd_bnd} — gerado, nao editar */
void __ESBMC_assume(_Bool c);
void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
    int x = nondet_int(), xd = nondet_int();
    int th = nondet_int(), thd = nondet_int();
    __ESBMC_assume(x   >= -{X_BND}  && x   <= {X_BND});
    __ESBMC_assume(xd  >= -{XD_BND} && xd  <= {XD_BND});
    __ESBMC_assume(th  >= -{TH_BND} && th  <= {TH_BND});
    __ESBMC_assume(thd >= -{thd_bnd} && thd <= {thd_bnd});

{corpo_controlador(w)}
{TANH}
    int th_acc  = (4040 * th - 375 * F_Q) / 256;
    int th_new  = th  + (5 * thd) / 256;
    int thd_new = thd + (5 * th_acc) / 256;
    int th_2    = th_new + (5 * thd_new) / 256;   /* F_Q alcanca theta aqui */

    __ESBMC_assert(th_2 >= -{TH_BND} && th_2 <= {TH_BND},
                   "PropB: theta sai da regiao segura apos 2 passos!");
    return 0;
}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    with open(PESOS) as fh:
        w = json.load(fh)

    tmp = os.path.join(AQUI, "_prop_b_tmp.c")
    print(f"{'|thd| <=':>9} {'rad/s':>7}  {'veredito':<9} {'tempo':>8}")
    achado = None
    for thd in (1280, 640, 320, 160, 80, 40, 20):
        with open(tmp, "w") as fh:
            fh.write(harness(w, thd))
        r = run_esbmc(tmp, timeout=a.timeout, no_unwinding_assertions=True,
                      boolector=True)
        print(f"{thd:>9} {thd/SCALE:>7.2f}  {r.status:<9} {r.time_taken:>7.1f}s")
        if r.status == SAFE and achado is None:
            achado = thd
            break
    os.remove(tmp)

    print()
    if achado:
        print(f"O controlador garante segurança de 2 passos para "
              f"|θ̇| <= {achado}/256 = {achado/SCALE:.2f} rad/s "
              f"({achado/SCALE*180/3.14159:.1f}°/s), com |θ| <= 12°.")
    else:
        print("Nenhum dos limites testados foi provado seguro. Isso é resultado, "
              "não falha do experimento: registre-o em vez de afrouxar a "
              "propriedade até ela passar.")


if __name__ == "__main__":
    main()
