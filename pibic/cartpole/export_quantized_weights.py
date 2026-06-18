"""
export_quantized_weights.py — Quantiza pesos do DDPG Actor para Q8.8 (scale=256).

Gera:
  - webapp/public/ddpg_weights_q88.json  (pesos inteiros para o browser)
  - quantization_report.json             (análise de erro float vs Q8.8)

O controlador quantizado no browser usa a mesma aritmética inteira que os
harnesses C verificados pelo ESBMC → zero gap de fidelidade.

Uso:
    python export_quantized_weights.py
"""

import json, os, math
import numpy as np
import torch

from ddpg_weight_extractor import extract_ddpg_weights
from cartpole_env import FORCE_MAX

HERE  = os.path.dirname(os.path.abspath(__file__))
SCALE = 256


def q(v: float) -> int:
    return int(round(v * SCALE))


def cdiv(a: int, b: int) -> int:
    return int(a / b)


def tanh_q88(z: int) -> int:
    """Piecewise linear tanh in Q8.8. 5 segments, max error ~3% vs real tanh.

    Key points:  tanh(0.25)=0.245  tanh(0.75)=0.635  tanh(1.5)=0.905  tanh(3.0)=0.995
    In Q8.8:     z=64→t=63         z=192→t=163       z=384→t=232      z=768→t=255
    """
    z_abs = abs(z)
    if z_abs <= 64:
        t = cdiv(z_abs * 252, 256)
    elif z_abs <= 192:
        t = 62 + cdiv((z_abs - 64) * 200, 256)
    elif z_abs <= 384:
        t = 162 + cdiv((z_abs - 192) * 92, 256)
    elif z_abs <= 768:
        t = 231 + cdiv((z_abs - 384) * 16, 256)
    else:
        t = 255
    return t if z >= 0 else -t


def forward_q88(state_q: list[int], qw: dict) -> int:
    h1 = []
    for i in range(len(qw["b1"])):
        pre = qw["b1"][i]
        for j in range(len(state_q)):
            pre += cdiv(state_q[j] * qw["w1"][i][j], SCALE)
        h1.append(max(0, pre))

    h2 = []
    for i in range(len(qw["b2"])):
        pre = qw["b2"][i]
        for j in range(len(h1)):
            pre += cdiv(h1[j] * qw["w2"][i][j], SCALE)
        h2.append(max(0, pre))

    z = qw["b_out"][0]
    for j in range(len(h2)):
        z += cdiv(h2[j] * qw["w_out"][0][j], SCALE)

    tanh_z = tanh_q88(z)
    return cdiv(tanh_z * 10 * SCALE, SCALE)


def forward_float(state: list[float], w: dict) -> float:
    import numpy as np
    x = np.array(state)
    h1 = np.maximum(0, np.array(w["w1"]) @ x + np.array(w["b1"]))
    h2 = np.maximum(0, np.array(w["w2"]) @ h1 + np.array(w["b2"]))
    z = (np.array(w["w_out"]) @ h2 + np.array(w["b_out"]))[0]
    return float(np.tanh(z) * FORCE_MAX)


def main():
    pth_path = os.path.join(HERE, "ddpg_actor_best.pth")
    print(f"Carregando pesos de {pth_path}")
    w_float = extract_ddpg_weights(pth_path)

    qw = {
        "w1":    [[q(v) for v in row] for row in w_float["w1"]],
        "b1":    [q(v) for v in w_float["b1"]],
        "w2":    [[q(v) for v in row] for row in w_float["w2"]],
        "b2":    [q(v) for v in w_float["b2"]],
        "w_out": [[q(v) for v in row] for row in w_float["w_out"]],
        "b_out": [q(v) for v in w_float["b_out"]],
    }

    out_path = os.path.join(HERE, "webapp", "public", "ddpg_weights_q88.json")
    with open(out_path, "w") as f:
        json.dump(qw, f, separators=(",", ":"))
    print(f"Pesos Q8.8 salvos em {out_path}")
    print(f"  w1:    {len(qw['w1'])}x{len(qw['w1'][0])}")
    print(f"  b1:    {len(qw['b1'])}")
    print(f"  w2:    {len(qw['w2'])}x{len(qw['w2'][0])}")
    print(f"  b2:    {len(qw['b2'])}")
    print(f"  w_out: {len(qw['w_out'])}x{len(qw['w_out'][0])}")
    print(f"  b_out: {len(qw['b_out'])}")

    # --- Análise de erro ---
    print("\n--- Análise de Erro: Float vs Q8.8 ---")
    np.random.seed(42)
    n_samples = 10000
    errors = []
    for _ in range(n_samples):
        state = [
            np.random.uniform(-2.4, 2.4),
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(-0.2094, 0.2094),
            np.random.uniform(-5.0, 5.0),
        ]
        f_float = forward_float(state, w_float)
        state_q = [q(s) for s in state]
        f_q88 = forward_q88(state_q, qw) / SCALE
        errors.append(abs(f_float - f_q88))

    errors = np.array(errors)
    report = {
        "scale": SCALE,
        "n_samples": n_samples,
        "max_abs_error_N": float(np.max(errors)),
        "mean_abs_error_N": float(np.mean(errors)),
        "p95_abs_error_N": float(np.percentile(errors, 95)),
        "p99_abs_error_N": float(np.percentile(errors, 99)),
        "force_range_N": [-FORCE_MAX, FORCE_MAX],
        "max_relative_error_pct": float(np.max(errors) / FORCE_MAX * 100),
    }

    report_path = os.path.join(HERE, "quantization_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  Amostras: {n_samples}")
    print(f"  Erro abs. máximo:  {report['max_abs_error_N']:.4f} N")
    print(f"  Erro abs. médio:   {report['mean_abs_error_N']:.4f} N")
    print(f"  Erro abs. p95:     {report['p95_abs_error_N']:.4f} N")
    print(f"  Erro abs. p99:     {report['p99_abs_error_N']:.4f} N")
    print(f"  Erro relativo max: {report['max_relative_error_pct']:.2f}%")
    print(f"\nRelatório salvo em {report_path}")


if __name__ == "__main__":
    main()
