"""Rodada canônica somente com os artefatos já exportados.

Não depende de PyTorch: usa o JSON Float32 exportado e o JSON Q8.8 exportado.
Os dois arquivos são identificados no manifesto da rodada; a geração oficial
de pesos a partir do checkpoint continua sendo uma etapa pendente.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


# .../pibic/evidencias/final_2026/scripts -> project root is parents[3].
ROOT = Path(__file__).resolve().parents[3]
EVID = ROOT / "evidencias" / "final_2026"
CART = ROOT / "cartpole"
FLOAT_PATH = CART / "webapp" / "public" / "ddpg_weights.json"
Q_PATH = CART / "webapp" / "public" / "ddpg_weights_q88.json"
OLD_REPORT = CART / "quantization_report.json"
OUT = EVID / "outputs" / "quantization_canonical.json"
CSV = EVID / "outputs" / "quantization_extremes.csv"
SCALE = 256
FORCE_MAX = 10.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def q(v: float) -> int:
    # Python round is ties-to-even, matching the exported implementation for
    # the non-tie values in this sample. Keep this explicit in the metadata.
    return int(round(float(v) * SCALE))


def cdiv(a: int, b: int) -> int:
    return int(a / b)


def tanh_q88(z: int) -> int:
    az = abs(z)
    if az <= 64:
        t = cdiv(az * 252, 256)
    elif az <= 192:
        t = 62 + cdiv((az - 64) * 200, 256)
    elif az <= 384:
        t = 162 + cdiv((az - 192) * 92, 256)
    elif az <= 768:
        t = 231 + cdiv((az - 384) * 16, 256)
    else:
        t = 255
    return t if z >= 0 else -t


def forward_float(state: list[float], w: dict) -> float:
    x = np.asarray(state, dtype=np.float64)
    h1 = np.maximum(0.0, np.asarray(w["net.0.weight"]) @ x + np.asarray(w["net.0.bias"]))
    h2 = np.maximum(0.0, np.asarray(w["net.2.weight"]) @ h1 + np.asarray(w["net.2.bias"]))
    z = float((np.asarray(w["net.4.weight"]) @ h2 + np.asarray(w["net.4.bias"]))[0])
    return float(np.tanh(z) * FORCE_MAX)


def forward_q88(state: list[float], qw: dict) -> tuple[float, int]:
    sq = [q(v) for v in state]
    h1 = []
    for i, bias in enumerate(qw["b1"]):
        pre = int(bias)
        for j, value in enumerate(sq):
            pre += cdiv(value * int(qw["w1"][i][j]), SCALE)
        h1.append(max(0, pre))
    h2 = []
    for i, bias in enumerate(qw["b2"]):
        pre = int(bias)
        for j, value in enumerate(h1):
            pre += cdiv(value * int(qw["w2"][i][j]), SCALE)
        h2.append(max(0, pre))
    z = int(qw["b_out"][0])
    for j, value in enumerate(h2):
        z += cdiv(value * int(qw["w_out"][0][j]), SCALE)
    t = tanh_q88(z)
    return cdiv(t * 10 * SCALE, SCALE) / SCALE, z


def main() -> None:
    w = json.loads(FLOAT_PATH.read_text(encoding="utf-8"))
    qw = json.loads(Q_PATH.read_text(encoding="utf-8"))
    rng = np.random.RandomState(42)
    errors: list[float] = []
    rows: list[dict] = []
    for i in range(10_000):
        state = [
            float(rng.uniform(-2.4, 2.4)),
            float(rng.uniform(-5.0, 5.0)),
            float(rng.uniform(-0.2094, 0.2094)),
            float(rng.uniform(-5.0, 5.0)),
        ]
        ff = forward_float(state, w)
        fq, z = forward_q88(state, qw)
        e = abs(ff - fq)
        errors.append(e)
        rows.append({"sample": i, "error_N": e, "float_N": ff, "q88_N": fq, "z_q88": z, "state": state})

    a = np.asarray(errors)
    order = np.argsort(a)[::-1][:20]
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "error_N", "float_N", "q88_N", "z_q88", "x", "x_dot", "theta", "theta_dot"])
        for ix in order:
            row = rows[int(ix)]
            writer.writerow([row["sample"], f"{row['error_N']:.12f}", f"{row['float_N']:.12f}", f"{row['q88_N']:.12f}", row["z_q88"], *[f"{v:.12f}" for v in row["state"]]])

    report = {
        "method": "reexecucao_deterministica_com_jsons_exportados",
        "seed": 42,
        "rng": "numpy.random.RandomState",
        "n_samples": int(len(a)),
        "scale": SCALE,
        "source_sha256": {str(p): sha256(p) for p in (FLOAT_PATH, Q_PATH, OLD_REPORT)},
        "recomputed": {
            "max_abs_error_N": float(np.max(a)),
            "mean_abs_error_N": float(np.mean(a)),
            "median_abs_error_N": float(np.median(a)),
            "p95_abs_error_N": float(np.percentile(a, 95)),
            "p99_abs_error_N": float(np.percentile(a, 99)),
            "max_relative_to_force_range_pct": float(np.max(a) / FORCE_MAX * 100),
        },
        "published_report": json.loads(OLD_REPORT.read_text(encoding="utf-8")),
        "interpretation": [
            "A mesma aritmetica Q8.8 foi reexecutada sobre os JSONs exportados.",
            "O erro maximo e reportado contra o Float32 de referencia; nao e uma divergencia entre harness e runtime Q8.8.",
            "O checkpoint PyTorch ainda precisa ser validado diretamente nesta rodada quando PyTorch estiver disponivel.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["recomputed"], indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")
    print(f"wrote {CSV}")


if __name__ == "__main__":
    main()
