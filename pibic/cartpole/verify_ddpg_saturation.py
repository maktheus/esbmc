"""Verificação corrigida de saturação do ator DDPG.

Cada harness calcula o grafo completo a partir do mesmo estado simbólico. Para
uma ReLU, provar ``pre >= 0`` significa que o neurônio está sempre ativo no
domínio (saturação); um contraexemplo ``pre < 0`` demonstra que há desativação.
Para a saída, os sinais de z são verificados com a rede real, sem ativações
independentes por intervalo.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVID = ROOT / "evidencias" / "final_2026"
QJSON = HERE / "webapp" / "public" / "ddpg_weights_q88.json"
ESBMC = ROOT / "QNNVerifier" / "esbmc-6.8.0" / "esbmc.exe"
SOLVER = "--z3"
TIMEOUT = int(os.environ.get("PIBITI_ESBMC_TIMEOUT", "8"))
X_BND, XD_BND, TH_BND, THD_BND = 614, 1280, 53, 1280


def load_weights() -> dict:
    with QJSON.open(encoding="utf-8") as f:
        return json.load(f)


def body(w: dict) -> str:
    lines = []
    for i in range(24):
        terms = " + ".join(f"(s{j}*({w['w1'][i][j]}))/256" for j in range(4))
        lines += [f"    int pre1_{i} = {terms} + ({w['b1'][i]});", f"    int h1_{i} = pre1_{i} > 0 ? pre1_{i} : 0;"]
    for i in range(24):
        terms = " + ".join(f"(h1_{j}*({w['w2'][i][j]}))/256" for j in range(24))
        lines += [f"    int pre2_{i} = {terms} + ({w['b2'][i]});", f"    int h2_{i} = pre2_{i} > 0 ? pre2_{i} : 0;"]
    terms = " + ".join(f"(h2_{j}*({w['w_out'][0][j]}))/256" for j in range(24))
    lines.append(f"    int z = {terms} + ({w['b_out'][0]});")
    return "\n".join(lines)


def make(name: str, assertion: str, w: dict) -> str:
    return f"""/* corrected saturation check: {name} */
void __ESBMC_assume(_Bool c); void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);
int main(void) {{
    int s0=nondet_int(), s1=nondet_int(), s2=nondet_int(), s3=nondet_int();
    __ESBMC_assume(s0 >= -{X_BND} && s0 <= {X_BND});
    __ESBMC_assume(s1 >= -{XD_BND} && s1 <= {XD_BND});
    __ESBMC_assume(s2 >= -{TH_BND} && s2 <= {TH_BND});
    __ESBMC_assume(s3 >= -{THD_BND} && s3 <= {THD_BND});
{body(w)}
{assertion}
    return 0;
}}
"""


def run(name: str, source: str) -> dict:
    hdir, ldir = EVID / "harnesses", EVID / "logs"
    hdir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)
    hp = hdir / f"{name}.c"
    sop, sep = ldir / f"{name}.stdout.log", ldir / f"{name}.stderr.log"
    hp.write_text(source, encoding="utf-8")
    cmd = [str(ESBMC), str(hp), "--no-unwinding-assertions", SOLVER]
    t0 = time.monotonic()
    timed_out = False
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        stdout, stderr = p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        stdout, stderr = e.stdout or "", e.stderr or ""
        timed_out = True
    raw = stdout + stderr
    if timed_out:
        status = "TIMEOUT"
    elif "VERIFICATION SUCCESSFUL" in raw:
        status = "SUCCESSFUL"
    elif "VERIFICATION FAILED" in raw:
        status = "FAILED"
    else:
        status = "UNKNOWN"
    sop.write_text(stdout, encoding="utf-8")
    sep.write_text(stderr, encoding="utf-8")
    ce = {}
    if status == "FAILED":
        for n in ("s0", "s1", "s2", "s3"):
            m = re.search(rf"\b{n}\s*=\s*(-?\d+)", raw)
            if m:
                ce[n] = int(m.group(1))
    return {"status": status, "elapsed_s": round(time.monotonic() - t0, 3), "timeout_s": TIMEOUT, "solver": SOLVER, "command": cmd, "harness": str(hp), "stdout": str(sop), "stderr": str(sep), "counterexample_state_q88": ce}


def main() -> None:
    w = load_weights()
    checks = {}
    for layer in (1, 2):
        for i in range(24):
            # Universal saturation: ReLU pre-activation is never negative.
            checks[f"layer_{layer}_{i}"] = run(f"corrected_sat_L{layer}_{i}", make(f"sat_L{layer}_{i}", f'    __ESBMC_assert(pre{layer}_{i} >= 0, "saturated_L{layer}_{i}");', w))

    # Full graph, no independent interval activations: sign responsiveness.
    checks["output_nonnegative_universal"] = run("corrected_sat_output_nonnegative", make("output_nonnegative", '    __ESBMC_assert(z >= 0, "output always nonnegative");', w))
    checks["output_nonpositive_universal"] = run("corrected_sat_output_nonpositive", make("output_nonpositive", '    __ESBMC_assert(z <= 0, "output always nonpositive");', w))

    summary = {}
    for layer in (1, 2):
        rs = [checks[f"layer_{layer}_{i}"] for i in range(24)]
        summary[f"layer_{layer}"] = {
            "total": 24,
            "saturated_proven": [i for i, r in enumerate(rs) if r["status"] == "SUCCESSFUL"],
            "not_saturated_counterexample": [i for i, r in enumerate(rs) if r["status"] == "FAILED"],
            "timeouts": [i for i, r in enumerate(rs) if r["status"] == "TIMEOUT"],
            "unknown": [i for i, r in enumerate(rs) if r["status"] == "UNKNOWN"],
        }
    summary["output"] = {
        "always_nonnegative": checks["output_nonnegative_universal"]["status"],
        "always_nonpositive": checks["output_nonpositive_universal"]["status"],
        "interpretation": "RESPONSIVE somente quando ambas as propriedades universais forem refutadas com contraexemplos validos; TIMEOUT/UNKNOWN e inconclusivo.",
    }
    out = EVID / "outputs" / "corrected_saturation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"method": "full_graph_symbolic_saturation", "weights": str(QJSON), "solver": SOLVER, "timeout_s": TIMEOUT, "summary": summary, "checks": checks}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
