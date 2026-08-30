"""Verificação corrigida de neurônios mortos do ator DDPG.

Esta versão usa o grafo Q8.8 completo, sem bounds artificiais de pré-ativação.
Na camada 2, a primeira camada é calculada no mesmo harness, de modo que as
ativações são ligadas ao mesmo estado simbólico. Os resultados da rodada são
gravados em evidencias/final_2026, não sobrescrevendo o JSON histórico.
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
SOLVER = "--z3"  # Boolector não está compilado no executável Windows disponível.
TIMEOUT = int(os.environ.get("PIBITI_ESBMC_TIMEOUT", "8"))
SCALE = 256
X_BND, XD_BND, TH_BND, THD_BND = 614, 1280, 53, 1280


def load_weights() -> dict:
    with QJSON.open(encoding="utf-8") as f:
        return json.load(f)


def controller_body(w: dict) -> str:
    lines = []
    for i in range(24):
        terms = " + ".join(f"(s{j}*({w['w1'][i][j]}))/256" for j in range(4))
        lines += [f"    int pre1_{i} = {terms} + ({w['b1'][i]});", f"    int h1_{i} = pre1_{i} > 0 ? pre1_{i} : 0;"]
    for i in range(24):
        terms = " + ".join(f"(h1_{j}*({w['w2'][i][j]}))/256" for j in range(24))
        lines += [f"    int pre2_{i} = {terms} + ({w['b2'][i]});", f"    int h2_{i} = pre2_{i} > 0 ? pre2_{i} : 0;"]
    return "\n".join(lines)


def harness(layer: int, index: int, w: dict) -> str:
    target = f"h{layer}_{index}"
    body = controller_body(w)
    return f"""/* corrected dead-neuron check: layer={layer}, neuron={index} */
void __ESBMC_assume(_Bool c); void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);
int main(void) {{
    int s0=nondet_int(), s1=nondet_int(), s2=nondet_int(), s3=nondet_int();
    __ESBMC_assume(s0 >= -{X_BND} && s0 <= {X_BND});
    __ESBMC_assume(s1 >= -{XD_BND} && s1 <= {XD_BND});
    __ESBMC_assume(s2 >= -{TH_BND} && s2 <= {TH_BND});
    __ESBMC_assume(s3 >= -{THD_BND} && s3 <= {THD_BND});
{body}
    __ESBMC_assert({target} == 0, "dead_neuron_L{layer}_{index}");
    return 0;
}}
"""


def parse_state(raw: str) -> dict:
    out = {}
    for name in ("s0", "s1", "s2", "s3"):
        m = re.search(rf"\b{name}\s*=\s*(-?\d+)", raw)
        if m:
            out[name] = int(m.group(1))
    return out


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
    return {"status": status, "elapsed_s": round(time.monotonic() - t0, 3), "timeout_s": TIMEOUT, "solver": SOLVER, "command": cmd, "harness": str(hp), "stdout": str(sop), "stderr": str(sep), "counterexample_state_q88": parse_state(raw) if status == "FAILED" else {}}


def main() -> None:
    w = load_weights()
    checks = {}
    for layer in (1, 2):
        for i in range(24):
            name = f"corrected_dead_L{layer}_{i}"
            checks[f"layer_{layer}_{i}"] = run(name, harness(layer, i, w))
    summary = {}
    for layer in (1, 2):
        rs = [checks[f"layer_{layer}_{i}"] for i in range(24)]
        summary[f"layer_{layer}"] = {
            "total": 24,
            "dead_proven": [i for i, r in enumerate(rs) if r["status"] == "SUCCESSFUL"],
            "active_counterexample": [i for i, r in enumerate(rs) if r["status"] == "FAILED"],
            "timeouts": [i for i, r in enumerate(rs) if r["status"] == "TIMEOUT"],
            "unknown": [i for i, r in enumerate(rs) if r["status"] == "UNKNOWN"],
        }
    out = EVID / "outputs" / "corrected_dead_neurons.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"method": "full_graph_symbolic_dead_neurons", "weights": str(QJSON), "solver": SOLVER, "timeout_s": TIMEOUT, "domain_q88": {"s0": [-X_BND, X_BND], "s1": [-XD_BND, XD_BND], "s2": [-TH_BND, TH_BND], "s3": [-THD_BND, THD_BND]}, "summary": summary, "checks": checks}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
