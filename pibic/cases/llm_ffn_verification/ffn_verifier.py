"""
FFN Formal Verifier — main orchestrator.

Pipeline
--------
1. Load ONNX model  →  extract FFN layer  →  slice to verification dimensions
2. Generate C harness with __ESBMC_assume / __ESBMC_assert
3. (Optional) Run Frama-C EVA to tighten interval bounds in generated C
4. Run ESBMC and parse result

Typical usage
-------------
# Verify layer 0 of a GPT-2 ONNX export, sliced to d_model=8, d_ff=32
python ffn_verifier.py gpt2.onnx --layer 0 --d-model-max 8 --d-ff-max 32

# Verify from raw numpy arrays (no ONNX needed)
python ffn_verifier.py --from-numpy --layer 0

# Smoke test: tiny hand-crafted FFN
python ffn_verifier.py --smoke-test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "core_verify"))

from onnx_ffn_extractor import FFNLayer, extract_ffn, extract_ffn_from_numpy
from ffn_c_generator import VerifConfig, generate_c

try:
    from esbmc_caller import run_esbmc, VerificationResult
    # Patch binary path to the bundled ESBMC if the build one is missing
    import esbmc_caller as _ec
    import os as _os
    if not _os.path.exists(_ec.ESBMC_BIN):
        _BUNDLED = str(_HERE.parent.parent / "QNNVerifier" / "esbmc-6.8.0" / "esbmc")
        if _os.path.exists(_BUNDLED):
            _ec.ESBMC_BIN = _BUNDLED
    _HAS_ESBMC_CALLER = True
except ImportError:
    _HAS_ESBMC_CALLER = False

try:
    from SMT_feedback_parser import FeedbackTrace
    _HAS_PARSER = True
except ImportError:
    _HAS_PARSER = False


# ---------------------------------------------------------------------------
# Smoke-test FFN (tiny, hand-crafted, analytically bounded)
# ---------------------------------------------------------------------------

def _make_smoke_test_layer(d_model: int = 2, d_ff: int = 4) -> FFNLayer:
    """
    Build a tiny FFN with known weights for smoke testing.

    W1 is xavier-uniform in [-0.5, 0.5]; b1=b2=0; W2 mirrors W1.
    With input in [-1, 1]:
      max |pre_act| ≤ d_model * 0.5 * 1 = 1.0
      GeLU(1.0) ≈ 0.841  →  max hidden ≈ 0.841
      max |output|  ≤ d_ff * 0.5 * 0.841 ≈ 1.68  (safe bound = 2.0)
    """
    rng = np.random.default_rng(42)
    W1 = rng.uniform(-0.5, 0.5, (d_ff, d_model)).astype(np.float32)
    b1 = np.zeros(d_ff, dtype=np.float32)
    W2 = rng.uniform(-0.5, 0.5, (d_model, d_ff)).astype(np.float32)
    b2 = np.zeros(d_model, dtype=np.float32)
    return FFNLayer(W1=W1, b1=b1, W2=W2, b2=b2, activation="gelu")


# ---------------------------------------------------------------------------
# Frama-C interval tightening (optional)
# ---------------------------------------------------------------------------

def _run_framac_intervals(c_path: str) -> Optional[str]:
    """
    Run Frama-C EVA to get tighter interval bounds and return them as a string.
    Returns None if Frama-C is not available.
    """
    import shutil
    import subprocess

    if not shutil.which("frama-c"):
        return None

    framac_cmd = ["frama-c", "-eva", c_path]
    try:
        result = subprocess.run(
            framac_cmd, capture_output=True, text=True, timeout=120
        )
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# ESBMC runner (fallback if core_verify not importable)
# ---------------------------------------------------------------------------

@dataclass
class _SimpleResult:
    is_safe: bool
    stdout: str
    stderr: str
    time_taken: float
    timeout_occurred: bool


def _find_esbmc_bin() -> str:
    """Locate ESBMC binary: system PATH → bundled copy."""
    import shutil
    if shutil.which("esbmc"):
        return "esbmc"
    bundled = str(_HERE.parent.parent / "QNNVerifier" / "esbmc-6.8.0" / "esbmc")
    if Path(bundled).exists():
        return bundled
    raise FileNotFoundError(
        "ESBMC binary not found. Install it or ensure the bundled copy is present at:\n"
        f"  {bundled}"
    )


def _run_esbmc_direct(
    c_path: str,
    d_model: int,
    d_ff: int,
    solver: str = "z3",
    timeout: int = 300,
) -> _SimpleResult:
    import subprocess

    unwind = max(d_ff, d_model) + 1
    esbmc_bin = _find_esbmc_bin()
    cmd = [
        esbmc_bin, c_path,
        "--floatbv",
        "--overflow-check",
        "--bounds-check",
        "--no-div-by-zero-check",
        "--unwind", str(unwind),
        "--no-unwinding-assertions",
        f"--{solver}",
    ]
    print("Running:", " ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - t0
        is_safe = (
            "VERIFICATION SUCCESSFUL" in proc.stdout
            or "VERIFICATION SUCCESSFUL" in proc.stderr
        )
        return _SimpleResult(
            is_safe=is_safe,
            stdout=proc.stdout,
            stderr=proc.stderr,
            time_taken=elapsed,
            timeout_occurred=False,
        )
    except subprocess.TimeoutExpired:
        return _SimpleResult(
            is_safe=False, stdout="", stderr="",
            time_taken=timeout, timeout_occurred=True,
        )


# ---------------------------------------------------------------------------
# Main verification entry point
# ---------------------------------------------------------------------------

def verify_ffn_layer(
    layer: FFNLayer,
    *,
    input_lb: float = -4.0,
    input_ub: float = 4.0,
    solver: str = "z3",
    timeout: int = 300,
    output_dir: str = ".",
    use_framac: bool = False,
) -> dict:
    """
    Full pipeline: generate C → (optionally run Frama-C) → run ESBMC.

    Returns a result dict with keys:
        is_safe, time_taken, timeout, violations, c_path, stdout, stderr
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    c_path = str(output_dir_path / f"ffn_layer{layer.source_layer_idx}_verify.c")
    import shutil
    for lut_name in ("gelu_lut.h", "silu_lut.h"):
        lut_src = _HERE / lut_name
        lut_dst = output_dir_path / lut_name
        if lut_src.exists() and not lut_dst.exists():
            shutil.copy(lut_src, lut_dst)

    # 1. Generate C harness
    cfg = VerifConfig(
        input_lb=input_lb,
        input_ub=input_ub,
        check_no_nan=True,
        check_no_inf=True,
        check_output_bound=True,
        gelu_lut_header="gelu_lut.h",
        unwind_hint=max(layer.d_ff, layer.d_model) + 1,
    )
    generate_c(layer, cfg, c_path)

    # 2. Optional Frama-C tightening
    if use_framac:
        print("Running Frama-C EVA for interval tightening …")
        framac_out = _run_framac_intervals(c_path)
        if framac_out:
            print("Frama-C intervals obtained (TODO: inject into C harness)")
        else:
            print("Frama-C not available, skipping.")

    # 3. Run ESBMC
    print(f"\nRunning ESBMC (solver={solver}, timeout={timeout}s) …")
    if _HAS_ESBMC_CALLER:
        result = run_esbmc(
            c_path,
            timeout=timeout,
            floatbv=True,
            overflow_check=True,
            bounds_check=True,
            unwind=max(layer.d_ff, layer.d_model) + 1,
            no_unwinding_assertions=True,
            **{solver: True},
        )
    else:
        result = _run_esbmc_direct(
            c_path, layer.d_model, layer.d_ff, solver=solver, timeout=timeout
        )

    # 4. Parse violations
    violations: list[str] = []
    if _HAS_PARSER and not result.is_safe:
        trace = FeedbackTrace(result.stdout + result.stderr)
        violations = trace.violations

    # 5. Summary
    status = "SAFE" if result.is_safe else ("TIMEOUT" if result.timeout_occurred else "UNSAFE")
    print(f"\n{'='*60}")
    print(f"  Result : {status}")
    print(f"  Time   : {result.time_taken:.1f}s")
    print(f"  Layer  : d_model={layer.d_model}, d_ff={layer.d_ff}")
    if violations:
        print(f"  Violations: {', '.join(violations)}")
    print(f"{'='*60}\n")

    return {
        "is_safe": result.is_safe,
        "time_taken": result.time_taken,
        "timeout": result.timeout_occurred,
        "violations": violations,
        "c_path": c_path,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Formally verify one FFN layer from an LLM using ESBMC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Smoke test (no model needed)
  python ffn_verifier.py --smoke-test

  # Verify layer 0 of exported GPT-2
  python ffn_verifier.py gpt2.onnx --layer 0 --d-model-max 8 --d-ff-max 32

  # Use bitwuzla solver, longer timeout
  python ffn_verifier.py model.onnx --solver bitwuzla --timeout 600
""",
    )
    p.add_argument("onnx", nargs="?", help="Path to ONNX model (optional if --smoke-test)")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run with a tiny hand-crafted FFN (no ONNX needed)")
    p.add_argument("--layer", type=int, default=0, help="FFN layer index (default: 0)")
    p.add_argument("--d-model-max", type=int, default=8,
                   help="Max d_model slice (default: 8)")
    p.add_argument("--d-ff-max", type=int, default=32,
                   help="Max d_ff slice (default: 32)")
    p.add_argument("--input-lb", type=float, default=-4.0,
                   help="Symbolic input lower bound (default: -4.0)")
    p.add_argument("--input-ub", type=float, default=4.0,
                   help="Symbolic input upper bound (default: 4.0)")
    p.add_argument("--solver", default="z3",
                   choices=["z3", "bitwuzla", "mathsat", "cvc4"],
                   help="SMT solver for ESBMC (default: z3)")
    p.add_argument("--timeout", type=int, default=300,
                   help="ESBMC timeout in seconds (default: 300)")
    p.add_argument("--output-dir", default="./verify_output",
                   help="Directory for generated files (default: ./verify_output)")
    p.add_argument("--use-framac", action="store_true",
                   help="Run Frama-C EVA for interval tightening before ESBMC")
    p.add_argument("--generate-only", action="store_true",
                   help="Only generate the C harness; do not run ESBMC")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.smoke_test:
        print("=== Smoke test: tiny 2×4 FFN ===")
        layer = _make_smoke_test_layer(d_model=2, d_ff=4)
    elif args.onnx:
        layer = extract_ffn(
            args.onnx,
            layer_idx=args.layer,
            d_model_max=args.d_model_max,
            d_ff_max=args.d_ff_max,
        )
    else:
        print("Error: provide an ONNX file or use --smoke-test")
        sys.exit(1)

    print(f"Layer: {layer}")

    if args.generate_only:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        c_path = str(out_dir / f"ffn_layer{layer.source_layer_idx}_verify.c")

        import shutil
        lut_src = _HERE / "gelu_lut.h"
        if lut_src.exists():
            shutil.copy(lut_src, out_dir / "gelu_lut.h")

        cfg = VerifConfig(
            input_lb=args.input_lb,
            input_ub=args.input_ub,
            gelu_lut_header="gelu_lut.h",
            unwind_hint=max(layer.d_ff, layer.d_model) + 1,
        )
        generate_c(layer, cfg, c_path)
        print(f"\nC harness generated: {c_path}")
        print(
            f"Run ESBMC manually:\n"
            f"  esbmc {c_path} --floatbv --overflow-check --bounds-check "
            f"--unwind {max(layer.d_ff, layer.d_model)+1} "
            f"--no-unwinding-assertions --{args.solver}"
        )
        return

    result = verify_ffn_layer(
        layer,
        input_lb=args.input_lb,
        input_ub=args.input_ub,
        solver=args.solver,
        timeout=args.timeout,
        output_dir=args.output_dir,
        use_framac=args.use_framac,
    )

    sys.exit(0 if result["is_safe"] else 1)


if __name__ == "__main__":
    main()
