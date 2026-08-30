"""Checks canônicos de reprodução dos dois contraexemplos DDPG.

Os harnesses são gerados nesta pasta e usam somente pesos Q8.8 exportados.
Eles não substituem uma prova universal: verificam que os estados publicados
de fato produzem a violação alegada no artefato quantizado.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVID = ROOT / "evidencias" / "final_2026"
CART = ROOT / "cartpole"
Q_PATH = CART / "webapp" / "public" / "ddpg_weights_q88.json"
ESBMC = CART / ".." / "QNNVerifier" / "esbmc-6.8.0" / "esbmc.exe"
HARNESS = EVID / "harnesses"
LOGS = EVID / "logs"
OUT = EVID / "outputs" / "concrete_checks.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def controller(qw: dict[str, list]) -> str:
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


def harness(name: str, state_q: tuple[int, int, int, int], assertion: str, extra: str = "") -> str:
    vals = "\n".join(f"    int s{i} = {v};" for i, v in enumerate(state_q))
    return f"""/* canonical concrete replay: {name} */
void __ESBMC_assert(_Bool c, const char *m);
int main(void) {{
{vals}
{controller(QW)}
{extra}
    __ESBMC_assert({assertion}, "{name}");
    return 0;
}}
"""


def run(name: str, source: str, timeout: int = 20) -> dict:
    c = HARNESS / f"{name}.c"
    so = LOGS / f"{name}.stdout.log"
    se = LOGS / f"{name}.stderr.log"
    c.write_text(source, encoding="utf-8")
    start = time.monotonic()
    try:
        proc = subprocess.run([str(ESBMC), str(c), "--no-unwinding-assertions", "--boolector"], capture_output=True, text=True, timeout=timeout)
        status = "SUCCESSFUL" if "VERIFICATION SUCCESSFUL" in proc.stdout + proc.stderr else ("FAILED" if "VERIFICATION FAILED" in proc.stdout + proc.stderr else "UNKNOWN")
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        proc = exc
        status = "TIMEOUT"
        timed_out = True
        proc.stdout = proc.stdout or ""
        proc.stderr = proc.stderr or ""
    elapsed = time.monotonic() - start
    so.write_text(proc.stdout, encoding="utf-8")
    se.write_text(proc.stderr, encoding="utf-8")
    return {"status": status, "timeout_s": timeout, "elapsed_s": round(elapsed, 3), "harness": str(c), "stdout": str(so), "stderr": str(se), "timed_out": timed_out}


def main() -> None:
    global QW
    QW = json.loads(Q_PATH.read_text(encoding="utf-8"))
    HARNESS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    # x=-0.0117, xd=-4.9219, theta=0.1094, theta_dot=0 -> q=[-3,-1260,28,0].
    right = run("property_a_right_replay", harness("property_a_right_replay", (-3, -1260, 28, 0), "z < 0"))
    # x=-0.7539, xd=-3.9219, theta=-0.1836, theta_dot=-1.5234 -> q=[-193,-1004,-47,-390].
    b_extra = f"""
{TANH}
    int th_acc = (4040 * s2 - 375 * F_Q) / 256;
    int th_new = s2 + (5 * s3) / 256;
"""
    safety = run("property_b_safety_replay", harness("property_b_safety_replay", (-193, -1004, -47, -390), "th_new < -53", b_extra))
    result = {
        "method": "ESBMC_concrete_replay",
        "esbmc": str(ESBMC),
        "weights_sha256": sha256(Q_PATH),
        "checks": {"property_a_right": right, "property_b_safety": safety},
        "interpretation": "SUCCESSFUL means ESBMC proved the explicit assertion for the fixed published state; it is a replay check, not a universal proof.",
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
