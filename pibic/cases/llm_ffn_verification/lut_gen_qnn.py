"""
Generate GeLU and SiLU LUT headers in QNNVerifier format.

QNNVerifier format (matching utils.h tanh/sigmoid LUTs):
  - Static float array
  - Integer index via: idx = int(x * 100) + CENTER_OFFSET
  - Boundary clamping handled in check_activation()

GeLU LUT: 1001 entries, x in [-5, 5], step=0.01
  idx = int(x * 100) + 500
  idx < 0    → return 0.0   (GeLU ≈ 0 for very negative x)
  idx >= 1001 → return x    (GeLU ≈ x for very positive x)

SiLU LUT: 1601 entries, x in [-8, 8], step=0.01
  idx = int(x * 100) + 800
  idx < 0    → return 0.0
  idx >= 1601 → return x

Usage:
    python lut_gen_qnn.py     # writes gelu_lut_qnn.h and silu_lut_qnn.h
"""

import math
from pathlib import Path


def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def _format_lut(values: list[float], per_row: int = 8) -> str:
    rows = []
    for i in range(0, len(values), per_row):
        rows.append("    " + ", ".join(f"{v:.6f}f" for v in values[i:i+per_row]))
    return ",\n".join(rows)


def generate_gelu_qnn(output_path: str = "gelu_lut_qnn.h") -> None:
    """
    GeLU LUT in QNNVerifier format.
    1001 entries covering x in [-5.0, 5.0] at step 0.01.
    Access: idx = fxp_to_int(fxp_mult(input, fxp_float_to_fxp(100.0f))) + 500
    """
    entries = [_gelu(-5.0 + i * 0.01) for i in range(1001)]
    body = _format_lut(entries)

    header = f"""\
#ifndef GELU_LUT_QNN_H
#define GELU_LUT_QNN_H

/*
 * GeLU Lookup Table — QNNVerifier format
 * 1001 entries, x ∈ [-5.0, 5.0], step = 0.01
 * GeLU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
 *
 * Index: idx = int(x * 100) + 500
 *   idx <  0    → 0.0f   (GeLU saturates near 0 for x << 0)
 *   idx >= 1001 → x      (GeLU ≈ x for x >> 0)
 *
 * Used by check_activation_gelu() in QNNVerifier-style harnesses.
 */
float lookupgelu[1001] = {{
{body}
}};

#endif /* GELU_LUT_QNN_H */
"""
    Path(output_path).write_text(header)
    print(f"GeLU LUT (QNN format) → {output_path}  (1001 entries, [-5, 5], step=0.01)")


def generate_silu_qnn(output_path: str = "silu_lut_qnn.h") -> None:
    """
    SiLU (Swish) LUT in QNNVerifier format.
    1601 entries covering x in [-8.0, 8.0] at step 0.01.
    Access: idx = fxp_to_int(fxp_mult(input, fxp_float_to_fxp(100.0f))) + 800
    """
    entries = [_silu(-8.0 + i * 0.01) for i in range(1601)]
    body = _format_lut(entries)

    header = f"""\
#ifndef SILU_LUT_QNN_H
#define SILU_LUT_QNN_H

/*
 * SiLU (Swish) Lookup Table — QNNVerifier format
 * 1601 entries, x ∈ [-8.0, 8.0], step = 0.01
 * SiLU(x) = x · σ(x) = x / (1 + exp(-x))
 * Used in: LLaMA, Mistral, Falcon, Phi
 *
 * Index: idx = int(x * 100) + 800
 *   idx <  0    → 0.0f
 *   idx >= 1601 → x
 */
float lookupsilu[1601] = {{
{body}
}};

#endif /* SILU_LUT_QNN_H */
"""
    Path(output_path).write_text(header)
    print(f"SiLU LUT (QNN format) → {output_path}  (1601 entries, [-8, 8], step=0.01)")


if __name__ == "__main__":
    generate_gelu_qnn("gelu_lut_qnn.h")
    generate_silu_qnn("silu_lut_qnn.h")
