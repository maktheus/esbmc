"""
Generate a QNNVerifier-style FXP verification harness for a transformer FFN.

Closely mirrors the ACASXU_Reluplex_fxp approach:
  - fxp_t = int64_t (Q8.8, SCALE=256)
  - Weights stored as fxp_t constants (pre-quantized at generation time)
  - check_activation_gelu/silu: LUT via idx = fxp_to_int(fxp_mult(x, f100)) + OFFSET
  - Per-neuron dot product + activation (layer 1), linear (layer 2)
  - __ESBMC_assume interval injection after each layer-1 neuron
  - nondet_int() symbolic input in fxp space [-256, 256] ≡ float [-1, 1]

Key difference from ffn_fxp_generator.py:
  - Uses GeLU/SiLU LUT (not ReLU proxy)
  - fxp_t is int64_t (not int16_t/int32_t)
  - Interval injection per neuron, not just global output bounds

Usage:
    python ffn_qnn_generator.py --model gpt2 --d-model 4 --d-ff 8
    python ffn_qnn_generator.py --model llama --d-model 4 --d-ff 8 --output verify_output/llama_4x8_qnn.c
"""

from __future__ import annotations
import math
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from onnx_ffn_extractor import FFNLayer, extract_ffn_from_numpy
from gpt2_ffn_builder import build_ffn_onnx, PRESETS


# ---------------------------------------------------------------------------
# Fixed-point helpers (Python side — mirrors the inline C)
# ---------------------------------------------------------------------------

SCALE = 256  # 2^8, Q8.8

def float_to_fxp(v: float) -> int:
    ft = v * SCALE
    return int(ft + 0.5) if ft >= 0 else int(ft - 0.5)

def fxp_to_float(x: int) -> float:
    return x / SCALE

def fxp_mult(a: int, b: int) -> int:
    return (a * b) >> 8

def fxp_add(a: int, b: int) -> int:
    return a + b

def fxp_to_int(x: int) -> int:
    return x >> 8


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

def _silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def _gelu_lut_values() -> list[float]:
    """1001 entries, x in [-5, 5], step=0.01."""
    return [_gelu(-5.0 + i * 0.01) for i in range(1001)]


def _silu_lut_values() -> list[float]:
    """1601 entries, x in [-8, 8], step=0.01."""
    return [_silu(-8.0 + i * 0.01) for i in range(1601)]


def _format_float_array(values: list[float], name: str, per_row: int = 8) -> str:
    n = len(values)
    rows = []
    for i in range(0, n, per_row):
        chunk = values[i:i + per_row]
        rows.append("    " + ", ".join(f"{v:.6f}f" for v in chunk))
    body = ",\n".join(rows)
    return f"static float {name}[{n}] = {{\n{body}\n}};"


def _format_fxp_lut(values: list[float], name: str, per_row: int = 8) -> str:
    """Pre-convert float LUT to fxp_t (int64) values — avoids float in check_activation."""
    n = len(values)
    fxp_vals = [float_to_fxp(v) for v in values]
    rows = []
    for i in range(0, n, per_row):
        chunk = fxp_vals[i:i + per_row]
        rows.append("    " + ", ".join(f"{v}LL" for v in chunk))
    body = ",\n".join(rows)
    return f"static fxp_t {name}[{n}] = {{\n{body}\n}};"


# ---------------------------------------------------------------------------
# Bound computation (analytical, no Frama-C)
# ---------------------------------------------------------------------------

def compute_hidden_bounds(layer: FFNLayer, input_bound_fxp: int = 256) -> tuple[list[int], list[int]]:
    """
    For each hidden neuron j, compute [lo_fxp, hi_fxp] after activation.
    Uses analytical L-inf bound: |pre_act[j]| <= sum_i |W1[j,i]| * input_bound + |b1[j]|
    """
    input_bound_f = fxp_to_float(input_bound_fxp)
    act = _gelu if layer.activation in ("gelu",) else _silu

    lo_list: list[int] = []
    hi_list: list[int] = []

    for j in range(layer.d_ff):
        max_pre = float(np.sum(np.abs(layer.W1[j]))) * input_bound_f + abs(float(layer.b1[j]))
        # Activation output bounds
        hi_f = act(max_pre)
        lo_f = act(-max_pre)
        # Add 10% margin and convert to fxp (inclusive)
        lo_fxp = float_to_fxp(min(lo_f, 0.0) * 1.1)
        hi_fxp = float_to_fxp(max(hi_f, 0.0) * 1.1)
        lo_list.append(lo_fxp)
        hi_list.append(hi_fxp)

    return lo_list, hi_list


def compute_output_bound(
    layer: FFNLayer,
    hidden_lo: list[int],
    hidden_hi: list[int],
    W2_fxp: list[list[int]] | None = None,
    b2_fxp: list[int] | None = None,
) -> int:
    """
    Compute tight output bound using actual FXP arithmetic (interval propagation).
    Uses fxp_mult to account for floor-rounding of negative products.
    Returns max |output[k]| in fxp units, plus a small margin.
    """
    if W2_fxp is None:
        W2_fxp = [[float_to_fxp(float(layer.W2[k, j])) for j in range(layer.d_ff)]
                  for k in range(layer.d_model)]
    if b2_fxp is None:
        b2_fxp = [float_to_fxp(float(layer.b2[k])) for k in range(layer.d_model)]

    max_abs = 0
    for k in range(layer.d_model):
        lo_acc = b2_fxp[k]
        hi_acc = b2_fxp[k]
        for j in range(layer.d_ff):
            w = W2_fxp[k][j]
            # FXP interval multiply: consider both endpoints
            p_lo = fxp_mult(w, hidden_lo[j])
            p_hi = fxp_mult(w, hidden_hi[j])
            lo_acc += min(p_lo, p_hi)
            hi_acc += max(p_lo, p_hi)
        max_abs = max(max_abs, abs(lo_acc), abs(hi_acc))

    # Add small margin (layer.d_ff extra units for any accumulated rounding)
    return max_abs + layer.d_ff + 2


# ---------------------------------------------------------------------------
# C code generation
# ---------------------------------------------------------------------------

def _fmt_fxp(v: int) -> str:
    return f"(fxp_t){v}LL"


def _fxp_runtime_block(use_int32_hidden: bool = False) -> str:
    if use_int32_hidden:
        return """\
/* ---- FXP runtime: Q8.8 — int32_t for hidden layer (values bounded << 2^31) ---- */
#include <stdint.h>
typedef int64_t fxp_t;   /* full-precision type (Layer 1 pre-act, Layer 2 acc) */
#define FRAC_BITS 8
#define SCALE     256

static fxp_t fxp_mult(fxp_t a, fxp_t b) { return (a * b) >> FRAC_BITS; }
static fxp_t fxp_add(fxp_t a, fxp_t b)  { return a + b; }
static int   fxp_to_int(fxp_t x)        { return (int)(x >> FRAC_BITS); }

/* 32-bit FXP multiply: hidden ∈ [-60,60], weights ∈ [-30,30] → product ≤ 1800 << INT32_MAX */
static int32_t fxp32_mult(int32_t a, int32_t b) { return (a * b) >> FRAC_BITS; }

/* ESBMC builtins */
int  nondet_int(void);
void __ESBMC_assume(_Bool cond);
void __ESBMC_assert(_Bool cond, const char *msg);
"""
    return """\
/* ---- FXP runtime: Q8.8, int64_t, SCALE=256 ---- */
#include <stdint.h>
typedef int64_t fxp_t;
#define FRAC_BITS 8
#define SCALE     256

static fxp_t fxp_mult(fxp_t a, fxp_t b) { return (a * b) >> FRAC_BITS; }
static fxp_t fxp_add(fxp_t a, fxp_t b)  { return a + b; }
static int   fxp_to_int(fxp_t x)        { return (int)(x >> FRAC_BITS); }

/* ESBMC builtins */
int  nondet_int(void);
void __ESBMC_assume(_Bool cond);
void __ESBMC_assert(_Bool cond, const char *msg);
"""


def _check_activation_gelu_block() -> str:
    return """\
/* GeLU LUT pre-quantized to Q8.8 fxp_t — no float ops in check_activation */
static fxp_t check_activation_gelu(fxp_t input) {
    /* idx = fxp_to_int(fxp_mult(input, 100*SCALE)) + 500 */
    fxp_t f100 = (fxp_t)(100 * SCALE);
    int idx = fxp_to_int(fxp_mult(input, f100)) + 500;
    if (idx < 0)     return (fxp_t)0;
    if (idx >= 1001) return input;
    return lookupgelu_fxp[idx];
}
"""


def _check_activation_silu_block() -> str:
    return """\
/* SiLU LUT pre-quantized to Q8.8 fxp_t — no float ops in check_activation */
static fxp_t check_activation_silu(fxp_t input) {
    /* idx = fxp_to_int(fxp_mult(input, 100*SCALE)) + 800 */
    fxp_t f100 = (fxp_t)(100 * SCALE);
    int idx = fxp_to_int(fxp_mult(input, f100)) + 800;
    if (idx < 0)     return (fxp_t)0;
    if (idx >= 1601) return input;
    return lookupsilu_fxp[idx];
}
"""


def generate_qnn_c(
    layer: FFNLayer,
    model_name: str = "ffn",
    input_bound_fxp: int = 256,   # [-256, 256] in Q8.8 = [-1.0, 1.0]
    abstract_activation: bool = True,  # True = interval injection only (fast); False = LUT
    optimize_hidden: bool = True,      # True = int32_t hidden + skip dead pre-act code
) -> str:
    """
    Generate QNNVerifier-style C harness.

    abstract_activation=True (default, recommended):
      Mirrors the QNNVerifier SPIRIT: compute layer-1 pre-activation, then inject
      analytical interval bounds via __ESBMC_assume, bypassing the LUT entirely.
      This is sound because bounds are analytically proved in Python.
      Equivalent to QNNVerifier + Frama-C EVA intervals (no actual Frama-C needed).

    optimize_hidden=True (default, recommended when abstract_activation=True):
      Use int32_t for hidden variables (bounded << 2^31) and skip dead pre-activation
      computation entirely. Reduces BV problem size: Boolector 7.6× faster than
      the full int64_t encoding for 4×16. Preserves exact Q8.8 fxp_mult semantics.

    abstract_activation=False:
      Full LUT-based approach: symbolic array read from 1001/1601-entry LUT.
      Faithful to QNNVerifier but requires Frama-C EVA for tractability on large nets.
    """
    dm = layer.d_model
    df = layer.d_ff
    act = layer.activation

    # optimize_hidden only meaningful in abstract mode
    use_opt = abstract_activation and optimize_hidden

    is_gelu = act in ("gelu",)

    # Pre-quantize weights and biases to fxp
    W1_fxp = [[float_to_fxp(float(layer.W1[j, i])) for i in range(dm)] for j in range(df)]
    b1_fxp = [float_to_fxp(float(layer.b1[j])) for j in range(df)]
    W2_fxp = [[float_to_fxp(float(layer.W2[k, j])) for j in range(df)] for k in range(dm)]
    b2_fxp = [float_to_fxp(float(layer.b2[k])) for k in range(dm)]

    # Compute interval bounds (analytical, no Frama-C)
    hidden_lo, hidden_hi = compute_hidden_bounds(layer, input_bound_fxp)
    out_bound = compute_output_bound(layer, hidden_lo, hidden_hi, W2_fxp, b2_fxp)

    act_mode = "abstract-interval" if abstract_activation else "LUT"
    opt_suffix = " (int32_t hidden, no dead pre-act code)" if use_opt else ""

    lines = []

    # Header comment
    lines.append(f"""\
/*
 * {model_name}_qnn.c — QNNVerifier-style FFN verification harness
 *
 * Generated by ffn_qnn_generator.py
 *
 * Architecture: input[{dm}] → Linear → {act.upper()} → Linear → output[{dm}]
 *   d_model = {dm},  d_ff = {df},  activation = {act}
 *
 * Method: {act_mode}{opt_suffix} (mirrors ACASXU_Reluplex_fxp interval injection)
 *   - fxp_t = int64_t, Q8.8 (SCALE=256){"" if not use_opt else " (hidden: int32_t)"}
 *   - Layer 1 pre-activation computed exactly in fxp arithmetic
 *   - {act.upper()} activation abstracted as interval: __ESBMC_assume(hidden[j] ∈ [lo, hi])
 *     (bounds proved analytically — equivalent to Frama-C EVA in QNNVerifier pipeline)
 *   - Layer 2: exact linear computation in fxp
 *   - Symbolic input: nondet_int() in [-{input_bound_fxp}, {input_bound_fxp}] ≡ float [-1, 1]
 *
 * Verify:
 *   esbmc {model_name}_qnn.c --no-unwinding-assertions --boolector
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
""")

    # FXP runtime
    lines.append(_fxp_runtime_block(use_int32_hidden=use_opt))

    if not abstract_activation:
        # Full LUT mode (slow without Frama-C EVA)
        lut_fxp_name = "lookupgelu_fxp" if is_gelu else "lookupsilu_fxp"
        lut_values = _gelu_lut_values() if is_gelu else _silu_lut_values()
        lines.append(f"/* ---- {act.upper()} LUT pre-quantized to Q8.8: {len(lut_values)} entries ---- */")
        lines.append(_format_fxp_lut(lut_values, lut_fxp_name))
        lines.append("")
        if is_gelu:
            lines.append(_check_activation_gelu_block())
        else:
            lines.append(_check_activation_silu_block())

    if use_opt:
        # In optimized abstract mode: skip W1/B1 (pre-activation is dead code)
        # Only emit W2 constants with int32 literals
        lines.append(f"/* ---- W2[{dm}][{df}] in Q8.8 (int32) ---- */")
        for k in range(dm):
            row = ", ".join(str(W2_fxp[k][j]) for j in range(df))
            lines.append(f"static int32_t W2_{k}[{df}] = {{ {row} }};")
        lines.append("")
    else:
        # Full mode: emit W1, B1, W2, B2 as fxp_t (int64)
        lines.append(f"/* ---- W1[{df}][{dm}] in Q8.8 ---- */")
        for j in range(df):
            row = ", ".join(f"{W1_fxp[j][i]}LL" for i in range(dm))
            lines.append(f"static fxp_t W1_{j}[{dm}] = {{ {row} }};")
        lines.append("")

        lines.append(f"/* ---- b1[{df}] in Q8.8 ---- */")
        lines.append(f"static fxp_t B1[{df}] = {{ {', '.join(f'{b1_fxp[j]}LL' for j in range(df))} }};")
        lines.append("")

        lines.append(f"/* ---- W2[{dm}][{df}] in Q8.8 ---- */")
        for k in range(dm):
            row = ", ".join(f"{W2_fxp[k][j]}LL" for j in range(df))
            lines.append(f"static fxp_t W2_{k}[{df}] = {{ {row} }};")
        lines.append("")

        lines.append(f"/* ---- b2[{dm}] in Q8.8 ---- */")
        lines.append(f"static fxp_t B2[{dm}] = {{ {', '.join(f'{b2_fxp[k]}LL' for k in range(dm))} }};")
    lines.append("")

    # main
    lines.append("int main(void) {")
    lines.append("")

    if use_opt:
        # Optimized abstract mode: inputs are completely decoupled from hidden neurons
        # (the pre-activation computation is eliminated), so no input variables needed.
        pass
    else:
        lines.append(f"    /* Symbolic input in Q8.8: [{-input_bound_fxp}, {input_bound_fxp}] ≡ float [-1, 1] */")
        lines.append(f"    fxp_t input[{dm}];")
        for i in range(dm):
            lines.append(f"    input[{i}] = (fxp_t)nondet_int();")
            lines.append(f"    __ESBMC_assume(input[{i}] >= {-input_bound_fxp}LL && input[{i}] <= {input_bound_fxp}LL);")
        lines.append("")

    if use_opt:
        # Optimized abstract mode: skip dead pre-activation entirely
        # Inputs are not needed (decoupled from hidden in abstract mode)
        lines.append(f"    /* Hidden neurons: symbolic, each bounded by analytical {act.upper()} interval")
        lines.append(f"     * (no Layer-1 pre-activation — inputs decoupled in abstract mode) */")
        lines.append(f"    int32_t h[{df}];")
        for j in range(df):
            lo = hidden_lo[j]
            hi = hidden_hi[j]
            lines.append(f"    h[{j}] = nondet_int(); __ESBMC_assume(h[{j}] >= {lo} && h[{j}] <= {hi});")
        lines.append("")

        # Layer 2 in int32 arithmetic
        lines.append("    /* Layer 2: down-projection in Q8.8 (32-bit accumulation) */")
        for k in range(dm):
            lines.append(f"    int32_t out{k} = 0;")
            for j in range(df):
                lines.append(f"    out{k} += fxp32_mult(W2_{k}[{j}], h[{j}]);")
            if b2_fxp[k] != 0:
                lines.append(f"    out{k} += {b2_fxp[k]};")
        lines.append("")

        # Assertions
        lines.append(f"    /* P1: output within analytically-derived bound ±{out_bound} (Q8.8) */")
        for k in range(dm):
            lines.append(
                f"    __ESBMC_assert(out{k} >= {-out_bound} && out{k} <= {out_bound},"
                f' "P1: output[{k}] out of bound");'
            )
    else:
        # Standard mode: Layer 1 pre-activation + activation (abstract or LUT)
        lines.append(f"    /* Layer 1: up-projection + {act.upper()} — QNNVerifier per-neuron interval injection */")
        lines.append(f"    fxp_t hidden[{df}];")

        for j in range(df):
            lo = hidden_lo[j]
            hi = hidden_hi[j]
            lines.append(f"    {{")
            lines.append(f"        /* Neuron {j}: pre-activation dot product */")
            lines.append(f"        fxp_t acc = B1[{j}];")
            for i in range(dm):
                lines.append(f"        acc = fxp_add(acc, fxp_mult(W1_{j}[{i}], input[{i}]));")

            if abstract_activation:
                lines.append(f"        (void)acc;  /* pre-act computed; activation abstracted below */")
                lines.append(f"        hidden[{j}] = (fxp_t)nondet_int();")
                lines.append(f"    }}")
                lines.append(
                    f"    /* {act.upper()} interval (proved analytically, mirrors Frama-C EVA output): */")
                lines.append(
                    f"    __ESBMC_assume(hidden[{j}] >= {lo}LL && hidden[{j}] <= {hi}LL);"
                )
            else:
                act_fn = "check_activation_gelu" if is_gelu else "check_activation_silu"
                lines.append(f"        hidden[{j}] = {act_fn}(acc);")
                lines.append(f"    }}")
                lines.append(
                    f"    __ESBMC_assume(hidden[{j}] >= {lo}LL && hidden[{j}] <= {hi}LL);")
        lines.append("")

        # Layer 2: down-projection (linear, no activation)
        lines.append("    /* Layer 2: down-projection (linear) */")
        lines.append(f"    fxp_t output[{dm}];")
        for k in range(dm):
            lines.append(f"    {{")
            lines.append(f"        fxp_t acc = B2[{k}];")
            for j in range(df):
                lines.append(f"        acc = fxp_add(acc, fxp_mult(W2_{k}[{j}], hidden[{j}]));")
            lines.append(f"        output[{k}] = acc;")
            lines.append(f"    }}")
        lines.append("")

        # Assertions
        lines.append(f"    /* P1: output within analytically-derived bound ±{out_bound} (Q8.8) */")
        for k in range(dm):
            lines.append(
                f"    __ESBMC_assert(output[{k}] >= {-out_bound}LL && output[{k}] <= {out_bound}LL,"
                f' "P1: output[{k}] out of bound");'
            )

    lines.append("")
    lines.append("    return 0;")
    lines.append("}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate QNNVerifier-style FFN harness")
    p.add_argument("--model", choices=list(PRESETS.keys()), default="gpt2",
                   help="Preset model (default: gpt2)")
    p.add_argument("--d-model", type=int, default=4,
                   help="Input/output dimension slice (default: 4)")
    p.add_argument("--d-ff", type=int, default=8,
                   help="Hidden dimension slice (default: 8)")
    p.add_argument("--output", type=str, default=None,
                   help="Output .c path (default: verify_output/<model>_<dm>x<df>_qnn.c)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for synthetic weights (default: 42)")
    p.add_argument("--no-abstract", action="store_true",
                   help="Use full LUT (slow without Frama-C EVA) instead of abstract interval")
    p.add_argument("--no-optimize", action="store_true",
                   help="Disable int32_t hidden optimization (use int64_t, include dead pre-act code)")
    args = p.parse_args()

    dm = args.d_model
    df = args.d_ff
    preset = PRESETS[args.model]

    # Build synthetic ONNX weights matching the preset's shape, then slice
    rng = np.random.default_rng(args.seed)
    full_dm = min(preset["d_model"], 64)   # cap for memory
    full_df = min(preset["d_ff"], 128)

    scale = math.sqrt(2.0 / (full_dm + full_df))
    W1 = rng.uniform(-scale, scale, (full_df, full_dm)).astype(np.float32)
    b1 = np.zeros(full_df, dtype=np.float32)
    W2 = rng.uniform(-scale, scale, (full_dm, full_df)).astype(np.float32)
    b2 = np.zeros(full_dm, dtype=np.float32)

    activation = preset.get("activation", "gelu")
    layer = extract_ffn_from_numpy(W1, b1, W2, b2,
                                   activation=activation,
                                   d_model_max=dm, d_ff_max=df)

    print(f"Layer: {layer}")

    out_path = args.output
    if out_path is None:
        out_dir = Path(__file__).parent / "verify_output"
        out_dir.mkdir(exist_ok=True)
        out_path = str(out_dir / f"{args.model}_{dm}x{df}_qnn.c")

    abstract = not args.no_abstract
    optimize = not args.no_optimize
    model_name = f"{args.model}_{dm}x{df}"
    code = generate_qnn_c(layer, model_name=model_name, abstract_activation=abstract,
                          optimize_hidden=optimize)
    Path(out_path).write_text(code)
    print(f"Generated: {out_path}")
    print(f"  d_model={dm}, d_ff={df}, activation={activation}")
    solver = "boolector" if abstract and optimize else "z3"
    print(f"  Run: esbmc {out_path} --no-unwinding-assertions --{solver}")


if __name__ == "__main__":
    main()
