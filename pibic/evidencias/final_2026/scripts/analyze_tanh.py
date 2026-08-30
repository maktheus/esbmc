"""Audita a aproximação de tanh usada pelo runtime/harness Q8.8."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVID = ROOT / "evidencias" / "final_2026"


def cdiv(a: int, b: int) -> int:
    return int(a / b)


def tanh_q88(z: int) -> int:
    a = abs(z)
    if a <= 64:
        t = cdiv(a * 252, 256)
    elif a <= 192:
        t = 62 + cdiv((a - 64) * 200, 256)
    elif a <= 384:
        t = 162 + cdiv((a - 192) * 92, 256)
    elif a <= 768:
        t = 231 + cdiv((a - 384) * 16, 256)
    else:
        t = 255
    return t if z >= 0 else -t


def main() -> None:
    max_abs = (-1.0, None)
    max_rel = (-1.0, None)
    for z in range(-10000, 10001):
        exact = math.tanh(z / 256.0)
        approx = tanh_q88(z) / 256.0
        abs_err = abs(approx - exact)
        rel_err = abs_err / abs(exact) if exact else 0.0
        if abs_err > max_abs[0]:
            max_abs = (abs_err, z)
        if rel_err > max_rel[0]:
            max_rel = (rel_err, z)
    out = {
        "z_domain_q88": [-10000, 10000],
        "max_absolute_error": max_abs[0],
        "max_absolute_error_pct_of_tanh_range": max_abs[0] * 100,
        "argmax_absolute_z_q88": max_abs[1],
        "max_relative_error_nonzero": max_rel[0],
        "argmax_relative_z_q88": max_rel[1],
        "note": "Relative error near zero is ill-conditioned; absolute error is the primary metric.",
    }
    path = EVID / "outputs" / "tanh_approximation.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
