"""Tenta duas propriedades universais em janela curta, sem alterar originais.

O resultado TIMEOUT é deliberadamente preservado como inconclusivo. Os bounds
de pré-ativação não são restringidos por constantes artificiais.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVID = ROOT / "evidencias" / "final_2026"
CART = ROOT / "cartpole"
Q_PATH = CART / "webapp" / "public" / "ddpg_weights_q88.json"
ESBMC = CART / ".." / "QNNVerifier" / "esbmc-6.8.0" / "esbmc.exe"


def controller(qw: dict) -> str:
    lines = []
    for i in range(24):
        terms = " + ".join(f"(s{j}*({qw['w1'][i][j]}))/256" for j in range(4))
        lines += [f"    int pre1_{i} = {terms} + ({qw['b1'][i]});", f"    int h1_{i} = pre1_{i} > 0 ? pre1_{i} : 0;"]
    for i in range(24):
        terms = " + ".join(f"(h1_{j}*({qw['w2'][i][j]}))/256" for j in range(24))
        lines += [f"    int pre2_{i} = {terms} + ({qw['b2'][i]});", f"    int h2_{i} = pre2_{i} > 0 ? pre2_{i} : 0;"]
    terms = " + ".join(f"(h2_{j}*({qw['w_out'][0][j]}))/256" for j in range(24))
    lines.append(f"    int z = {terms} + ({qw['b_out'][0]});")
    return "\n".join(lines)


TANH = """
    int z_abs = z >= 0 ? z : -z;
    int tanh_abs;
    if (z_abs <= 64)        tanh_abs = (z_abs * 252) / 256;
    else if (z_abs <= 192)  tanh_abs = 62 + ((z_abs - 64) * 200) / 256;
    else if (z_abs <= 384)  tanh_abs = 162 + ((z_abs - 192) * 92) / 256;
    else if (z_abs <= 768)  tanh_abs = 231 + ((z_abs - 384) * 16) / 256;
    else                    tanh_abs = 255;
    int tanh_z = z >= 0 ? tanh_abs : -tanh_abs;
    int F_Q = (tanh_z * 10 * 256) / 256;
""".strip("\n")


def make(name: str, assertion: str) -> str:
    return f"""/* canonical short universal check: {name} */
void __ESBMC_assume(_Bool c); void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);
int main(void) {{
    int s0=nondet_int(), s1=nondet_int(), s2=nondet_int(), s3=nondet_int();
    __ESBMC_assume(s0 >= -614 && s0 <= 614);
    __ESBMC_assume(s1 >= -1280 && s1 <= 1280);
    __ESBMC_assume(s2 >= -53 && s2 <= 53);
    __ESBMC_assume(s3 >= -1280 && s3 <= 1280);
{controller(QW)}
{TANH}
    __ESBMC_assert({assertion}, "{name}");
    return 0;
}}
"""


def run(name: str, src: str, solver: str, timeout: int = 12) -> dict:
    h = EVID / "harnesses" / f"{name}.c"
    so = EVID / "logs" / f"{name}.stdout.log"
    se = EVID / "logs" / f"{name}.stderr.log"
    h.write_text(src, encoding="utf-8")
    start = time.monotonic()
    try:
        p = subprocess.run([str(ESBMC), str(h), "--no-unwinding-assertions", solver], capture_output=True, text=True, timeout=timeout)
        raw = p.stdout + p.stderr
        status = "SUCCESSFUL" if "VERIFICATION SUCCESSFUL" in raw else ("FAILED" if "VERIFICATION FAILED" in raw else "UNKNOWN")
        timed_out = False
    except subprocess.TimeoutExpired as e:
        p = e
        p.stdout = p.stdout or ""
        p.stderr = p.stderr or ""
        status = "TIMEOUT"
        timed_out = True
    so.write_text(p.stdout, encoding="utf-8")
    se.write_text(p.stderr, encoding="utf-8")
    return {"status": status, "timeout_s": timeout, "elapsed_s": round(time.monotonic() - start, 3), "harness": str(h), "stdout": str(so), "stderr": str(se), "timed_out": timed_out}


def main() -> None:
    global QW
    QW = json.loads(Q_PATH.read_text(encoding="utf-8"))
    checks = {
        "property_c_bounds_boolector": run("property_c_bounds_boolector", make("property_c_bounds_boolector", "F_Q >= -2560 && F_Q <= 2560"), "--boolector"),
        "property_c_bounds_z3": run("property_c_bounds_z3", make("property_c_bounds_z3", "F_Q >= -2560 && F_Q <= 2560"), "--z3"),
        "property_a_right_boolector": run("property_a_right_boolector", make("property_a_right_boolector", "!(s2 > 25 && s3 >= 0) || z > 0"), "--boolector"),
        "property_a_right_z3": run("property_a_right_z3", make("property_a_right_z3", "!(s2 > 25 && s3 >= 0) || z > 0"), "--z3"),
    }
    out = EVID / "outputs" / "short_universal_checks.json"
    out.write_text(json.dumps({"method": "ESBMC_short_universal", "checks": checks, "interpretation": "TIMEOUT/UNKNOWN e inconclusivo; nenhum resultado e convertido em sucesso ou falha da propriedade."}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
