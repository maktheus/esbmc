"""
Generate a self-contained C verification harness for one transformer FFN layer.

The generated file is ready to pass directly to ESBMC:
    esbmc output.c --floatbv --overflow-check --bounds-check \\
                   --unwind <D_FF> --no-unwinding-assertions --z3

Design choices
--------------
* Uses float arithmetic with ESBMC's --floatbv IEEE-754 encoding.
  (Avoids the complexity of fixed-point; quantised variant can be added later.)
* Weights are hardcoded as float arrays (same pattern as QNNVerifier benchmarks).
* Inputs are symbolic via nondet_float() constrained with __ESBMC_assume.
* GeLU is approximated via a compile-time lookup table (gelu_lut.h).
* Properties verified (selectable):
    1. no_nan    — no NaN in hidden layer or output
    2. no_inf    — no +/-Inf
    3. output_bound — output[k] in [out_lb, out_ub] for all k  (requires bound estimate)
    4. bounds_check — array out-of-bounds (automatic via --bounds-check flag)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import textwrap

import numpy as np

from onnx_ffn_extractor import FFNLayer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VerifConfig:
    """Controls what the generated C harness checks."""

    # Symbolic input range: every input[i] ∈ [input_lb, input_ub]
    input_lb: float = -4.0
    input_ub: float = 4.0

    # Properties to emit __ESBMC_assert for
    check_no_nan: bool = True
    check_no_inf: bool = True
    check_output_bound: bool = True

    # Output bound (computed automatically if None)
    output_lb: Optional[float] = None
    output_ub: Optional[float] = None

    # Path to gelu_lut.h (relative or absolute)
    gelu_lut_header: str = "gelu_lut.h"

    # ESBMC loop unwind hint (printed as a comment; pass via CLI)
    unwind_hint: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_float(v: float) -> str:
    """Format float as C float literal with 7 significant digits."""
    if np.isnan(v):
        return "0.0f /*NaN replaced*/"
    return f"{v:.7g}f"


def _array_1d_literal(arr: np.ndarray, indent: int = 4) -> str:
    pad = " " * indent
    entries = [_fmt_float(float(v)) for v in arr]
    rows = []
    for i in range(0, len(entries), 6):
        rows.append(pad + ", ".join(entries[i : i + 6]))
    return "{\n" + ",\n".join(rows) + "\n}"


def _array_2d_literal(arr: np.ndarray, indent: int = 4) -> str:
    """Emit a 2-D float array as a nested C literal.  arr shape: [rows, cols]."""
    pad = " " * indent
    rows_out = []
    for row in arr:
        entries = [_fmt_float(float(v)) for v in row]
        rows_out.append(pad + "{ " + ", ".join(entries) + " }")
    return "{\n" + ",\n".join(rows_out) + "\n}"


# ---------------------------------------------------------------------------
# Code generator
# ---------------------------------------------------------------------------

def generate_c(
    layer: FFNLayer,
    cfg: VerifConfig,
    output_path: str,
) -> str:
    """
    Generate a C verification harness for *layer* and write it to *output_path*.

    Returns the generated C source as a string.
    """
    D_MODEL = layer.d_model
    D_FF    = layer.d_ff

    # Compute automatic output bound if not provided
    out_mag = layer.max_output_magnitude(input_bound=max(abs(cfg.input_lb), abs(cfg.input_ub)))
    # Add 20 % safety margin
    auto_bound = out_mag * 1.2 + 1.0

    out_lb = cfg.output_lb if cfg.output_lb is not None else -auto_bound
    out_ub = cfg.output_ub if cfg.output_ub is not None else  auto_bound

    unwind = cfg.unwind_hint if cfg.unwind_hint is not None else max(D_FF, D_MODEL) + 1

    # ---- Weight literals ------------------------------------------------
    W1_lit = _array_2d_literal(layer.W1)
    b1_lit = _array_1d_literal(layer.b1)
    W2_lit = _array_2d_literal(layer.W2)
    b2_lit = _array_1d_literal(layer.b2)

    # ---- Assertion blocks -----------------------------------------------
    assert_hidden = ""
    if cfg.check_no_nan:
        assert_hidden += f"""
    /* Property: no NaN in hidden pre-GeLU activations */
    for (int j = 0; j < D_FF; j++) {{
        __ESBMC_assert(pre_act[j] == pre_act[j], "NaN in hidden pre-activation");
    }}"""
    if cfg.check_no_inf:
        assert_hidden += f"""
    /* Property: no Inf in hidden pre-GeLU activations */
    for (int j = 0; j < D_FF; j++) {{
        __ESBMC_assert(pre_act[j] < 1e38f && pre_act[j] > -1e38f, "Inf in hidden pre-activation");
    }}"""

    assert_output = ""
    if cfg.check_no_nan:
        assert_output += f"""
    /* Property: no NaN in FFN output */
    for (int k = 0; k < D_MODEL; k++) {{
        __ESBMC_assert(output[k] == output[k], "NaN in FFN output");
    }}"""
    if cfg.check_no_inf:
        assert_output += f"""
    /* Property: no Inf in FFN output */
    for (int k = 0; k < D_MODEL; k++) {{
        __ESBMC_assert(output[k] < 1e38f && output[k] > -1e38f, "Inf in FFN output");
    }}"""
    if cfg.check_output_bound:
        assert_output += f"""
    /* Property: output stays within computed bound [{out_lb:.4f}, {out_ub:.4f}]
     * (derived from weight L-inf norms + input bound {cfg.input_lb} .. {cfg.input_ub}) */
    for (int k = 0; k < D_MODEL; k++) {{
        __ESBMC_assert(output[k] >= {_fmt_float(out_lb)}, "FFN output below lower bound");
        __ESBMC_assert(output[k] <= {_fmt_float(out_ub)}, "FFN output above upper bound");
    }}"""

    # ---- Full C source --------------------------------------------------
    src = f"""\
/*
 * FFN Verification Harness — auto-generated by ffn_c_generator.py
 *
 * Layer index : {layer.source_layer_idx}
 * Dimensions  : d_model={D_MODEL}, d_ff={D_FF}
 * Activation  : {layer.activation.upper()} (via LUT)
 * Input range : [{cfg.input_lb}, {cfg.input_ub}]
 * Output bound: [{out_lb:.4f}, {out_ub:.4f}]
 *
 * Verify with:
 *   esbmc {Path(output_path).name} \\
 *       --floatbv --overflow-check --bounds-check \\
 *       --unwind {unwind} --no-unwinding-assertions \\
 *       --z3
 *
 * Properties checked:
 *   - No NaN/Inf in hidden or output activations
 *   - Output bounded in [{out_lb:.4f}, {out_ub:.4f}]
 *   - Array accesses in bounds (via --bounds-check flag)
 */

#include <math.h>
#include "{cfg.gelu_lut_header}"

/* ---- Dimensions --------------------------------------------------------- */
#define D_MODEL {D_MODEL}
#define D_FF    {D_FF}

/* ---- ESBMC builtins ----------------------------------------------------- */
float nondet_float(void);
void  __ESBMC_assume(_Bool cond);
void  __ESBMC_assert(_Bool cond, const char *msg);

/* ---- Weights (sliced from model, layer {layer.source_layer_idx}) --------- */

/* Up-projection W1[D_FF][D_MODEL] */
static float W1[D_FF][D_MODEL] = {W1_lit};

/* Up-projection bias b1[D_FF] */
static float b1[D_FF] = {b1_lit};

/* Down-projection W2[D_MODEL][D_FF] */
static float W2[D_MODEL][D_FF] = {W2_lit};

/* Down-projection bias b2[D_MODEL] */
static float b2[D_MODEL] = {b2_lit};

/* ---- Verification entry point ------------------------------------------- */
int main(void) {{

    /* --- Symbolic input --------------------------------------------------- */
    float input[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {{
        input[i] = nondet_float();
        __ESBMC_assume(input[i] >= {_fmt_float(cfg.input_lb)});
        __ESBMC_assume(input[i] <= {_fmt_float(cfg.input_ub)});
    }}

    /* --- Layer 1: up-projection + GeLU ------------------------------------ */
    /* pre_act[j] = W1[j] · input + b1[j]  */
    float pre_act[D_FF];
    for (int j = 0; j < D_FF; j++) {{
        float acc = b1[j];
        for (int i = 0; i < D_MODEL; i++) {{
            acc += W1[j][i] * input[i];
        }}
        pre_act[j] = acc;
    }}
{assert_hidden}
    /* Apply GeLU via lookup table */
    float hidden[D_FF];
    for (int j = 0; j < D_FF; j++) {{
        hidden[j] = geluLUT(pre_act[j]);
    }}

    /* --- Layer 2: down-projection ----------------------------------------- */
    /* output[k] = W2[k] · hidden + b2[k]  */
    float output[D_MODEL];
    for (int k = 0; k < D_MODEL; k++) {{
        float acc = b2[k];
        for (int j = 0; j < D_FF; j++) {{
            acc += W2[k][j] * hidden[j];
        }}
        output[k] = acc;
    }}
{assert_output}
    return 0;
}}
"""
    Path(output_path).write_text(src)
    print(
        f"C harness written → {output_path}\n"
        f"  d_model={D_MODEL}, d_ff={D_FF}, unwind≥{unwind}\n"
        f"  output bound: [{out_lb:.4f}, {out_ub:.4f}]"
    )
    return src


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(
        description="Generate ESBMC C harness for an FFN layer extracted from ONNX"
    )
    p.add_argument("onnx", help="Path to .onnx model file")
    p.add_argument("--layer", type=int, default=0, help="FFN layer index (default: 0)")
    p.add_argument("--d-model-max", type=int, default=8,
                   help="Max d_model slice for verification (default: 8)")
    p.add_argument("--d-ff-max", type=int, default=32,
                   help="Max d_ff slice for verification (default: 32)")
    p.add_argument("--input-lb", type=float, default=-4.0)
    p.add_argument("--input-ub", type=float, default=4.0)
    p.add_argument("--output", default="ffn_verify.c", help="Output C file (default: ffn_verify.c)")
    p.add_argument("--no-nan-check", action="store_true")
    p.add_argument("--no-inf-check", action="store_true")
    p.add_argument("--no-bound-check", action="store_true")
    args = p.parse_args()

    from onnx_ffn_extractor import extract_ffn

    layer = extract_ffn(
        args.onnx,
        layer_idx=args.layer,
        d_model_max=args.d_model_max,
        d_ff_max=args.d_ff_max,
    )
    cfg = VerifConfig(
        input_lb=args.input_lb,
        input_ub=args.input_ub,
        check_no_nan=not args.no_nan_check,
        check_no_inf=not args.no_inf_check,
        check_output_bound=not args.no_bound_check,
    )
    generate_c(layer, cfg, args.output)
    print(f"\nRun ESBMC:\n  esbmc {args.output} --floatbv --overflow-check --bounds-check "
          f"--unwind {max(args.d_ff_max, args.d_model_max)+1} --no-unwinding-assertions --z3")
