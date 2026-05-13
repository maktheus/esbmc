# LLM FFN Formal Verification

Formal verification of a single **Feed-Forward Network (FFN)** layer extracted from a Large Language Model deploy, using ESBMC (SMT-based Model Checker).

## What this verifies

A transformer FFN block has the shape:

```
input[d_model]
    └─ Linear(d_model → d_ff)  +  GeLU / SiLU
         └─ Linear(d_ff → d_model)
              └─ output[d_model]
```

For GPT-2: `d_model=768, d_ff=3072`.  
For LLaMA-7B: `d_model=4096, d_ff=11008`.

Because verifying the full dimensions is computationally intractable, the pipeline **slices** the layer to a tractable size (e.g., `d_model=4, d_ff=8`) and proves properties on that slice using real model weights.

### Properties proved (for all possible inputs in the given range)

| # | Property | How |
|---|-----------|-----|
| P1 | No integer overflow in matrix multiply | `--overflow-check` flag |
| P2 | Hidden pre-activations within analytical bound | `__ESBMC_assert` |
| P3 | Output within analytically-derived bound | `__ESBMC_assert` |
| P4 | No array out-of-bounds accesses | ESBMC default behaviour |

---

## Architecture

Two generators are provided, differing in how they handle the non-linear activation:

### Method 1 — Fixed-point ReLU proxy (`ffn_fxp_generator.py`)

```
gpt2_ffn_builder.py          # Build ONNX from scratch (no torch needed)
        │
        ▼  (gpt2_ffn.onnx / llama_ffn.onnx)
        │
onnx_ffn_extractor.py        # Parse ONNX → extract W1, b1, W2, b2
        │
        ├─── ffn_c_generator.py     # Float harness  (--floatbv, slow for large dims)
        │          └─ gelu_lut.h / silu_lut.h
        │
        └─── ffn_fxp_generator.py   # Fixed-point Q8.8 harness (ReLU proxy)
                   └─ mini_ffn_fxp_test.c (smoke test)
                        │
                        ▼
              ffn_verifier.py       # Orchestrator: extract → generate → ESBMC
                        │
                        ▼
              esbmc_caller.py       # Subprocess wrapper (../../../core_verify/)
```

Activation: GELU/SiLU replaced by ReLU proxy. Fast for small dimensions, but
the proxy introduces soundness approximation.

### Method 2 — QNNVerifier abstract-interval injection (`ffn_qnn_generator.py`)

```
gpt2_ffn_builder.py          # same ONNX builder
        │
onnx_ffn_extractor.py        # same extractor
        │
        └─── ffn_qnn_generator.py   # QNNVerifier-style harness  ← exact activation
                   └─ lut_gen_qnn.py / gelu_lut_qnn.h / silu_lut_qnn.h
```

Mirrors the **ACASXU_Reluplex_fxp** technique from QNNVerifier:
- `fxp_t = int64_t`, Q8.8 (SCALE=256)
- Layer 1 pre-activation computed exactly in FXP arithmetic
- GELU/SiLU replaced by **abstract nondet + interval assumption**:
  `hidden[j] = nondet_int(); __ESBMC_assume(hidden[j] ∈ [lo, hi])`
  where `[lo, hi]` is analytically derived (equivalent to Frama-C EVA output)
- Layer 2: exact linear FXP computation
- Output bounded by interval arithmetic using actual `fxp_mult` floor-rounding

---

## Files

| File | Purpose |
|------|---------|
| `gpt2_ffn_builder.py` | Build GPT-2 / LLaMA / TinyLlama FFN as ONNX (no torch) |
| `onnx_ffn_extractor.py` | Extract FFN weights from any ONNX transformer model |
| `ffn_c_generator.py` | Generate float C harness (good for tiny dims) |
| `ffn_fxp_generator.py` | Generate fixed-point Q8.8 C harness with ReLU proxy |
| `ffn_qnn_generator.py` | Generate QNNVerifier-style Q8.8 harness with abstract activation **(exact)** |
| `ffn_verifier.py` | End-to-end orchestrator CLI |
| `gelu_lut_gen.py` | Generate GeLU LUT header (1000 entries, [-5, 5]) |
| `silu_lut_gen.py` | Generate SiLU LUT header (1600 entries, [-8, 8]) |
| `lut_gen_qnn.py` | Generate pre-quantized fxp_t LUT headers for QNN method |
| `gelu_lut.h` | Pre-generated GeLU LUT (float values) |
| `silu_lut.h` | Pre-generated SiLU LUT (float values) |
| `gelu_lut_qnn.h` | Pre-generated GeLU LUT (fxp_t Q8.8 values, for QNN method) |
| `silu_lut_qnn.h` | Pre-generated SiLU LUT (fxp_t Q8.8 values, for QNN method) |
| `mini_ffn_smoke_test.c` | Smoke test: 2×4×2, float, GeLU approx |
| `mini_ffn_fxp_test.c` | Smoke test: 2×4×2, fixed-point Q8.8, ReLU |
| `verify_output/` | Generated C harnesses from real model weights |
| `.gitignore` | Excludes `*.onnx` (large generated artifacts) |

---

## Quick start

### 1. Smoke test (no model needed, runs in < 1 second)

```bash
esbmc mini_ffn_fxp_test.c \
    --overflow-check \
    --unwind 5 --no-unwinding-assertions \
    --z3
# → VERIFICATION SUCCESSFUL (0.25 s)
```

### 2. Verify GPT-2 layer 0 (4×8 slice, ~12 seconds) — Method 1

```bash
# Generate model ONNX
python gpt2_ffn_builder.py --model gpt2 --out gpt2_ffn.onnx

# Generate fixed-point harness
python ffn_fxp_generator.py gpt2_ffn.onnx \
    --layer 0 --d-model-max 4 --d-ff-max 8 \
    --out verify_output/gpt2_4x8.c

# Verify
esbmc verify_output/gpt2_4x8.c \
    --overflow-check \
    --unwind 9 --no-unwinding-assertions \
    --z3
# → VERIFICATION SUCCESSFUL (~12 s)
```

### 3. Verify LLaMA-7B layer 0 (4×8 slice, ~9 seconds) — Method 1

```bash
python gpt2_ffn_builder.py --model llama-7b --out llama_ffn.onnx

python ffn_fxp_generator.py llama_ffn.onnx \
    --layer 0 --d-model-max 4 --d-ff-max 8 \
    --out verify_output/llama_4x8.c

esbmc verify_output/llama_4x8.c \
    --overflow-check \
    --unwind 9 --no-unwinding-assertions \
    --z3
# → VERIFICATION SUCCESSFUL (~9 s)
```

### 6. QNNVerifier-style abstract activation — Method 2 (exact GeLU/SiLU)

```bash
# Generate QNN abstract harness (int32_t hidden, Boolector-optimized)
python ffn_qnn_generator.py --model gpt2 --d-model 4 --d-ff 8

# Verify
esbmc verify_output/gpt2_4x8_qnn.c --no-unwinding-assertions --boolector
# → VERIFICATION SUCCESSFUL (~11 s)

# Larger case
python ffn_qnn_generator.py --model gpt2 --d-model 4 --d-ff 16
esbmc verify_output/gpt2_4x16_qnn.c --no-unwinding-assertions --boolector
# → VERIFICATION SUCCESSFUL (~30 s)
```

The QNN method proves exact GeLU/SiLU semantics by abstracting each hidden neuron
as a symbolic interval derived analytically, eliminating the ReLU-proxy soundness gap.
Key optimizations: `int32_t` hidden variables, skip dead pre-activation code, Boolector solver.

### 4. End-to-end via orchestrator

```bash
python ffn_verifier.py gpt2_ffn.onnx \
    --layer 0 --d-model-max 4 --d-ff-max 8 \
    --solver z3 --timeout 60
```

### 5. Generate only the C harness (manual ESBMC run)

```bash
python ffn_verifier.py gpt2_ffn.onnx --generate-only \
    --d-model-max 4 --d-ff-max 8
# Prints the exact esbmc command to run
```

---

## Supported models

| Model | Preset flag | d_model | d_ff | Activation |
|-------|-------------|---------|------|------------|
| GPT-2 small | `--model gpt2` | 768 | 3072 | GeLU |
| GPT-2 medium | `--model gpt2-medium` | 1024 | 4096 | GeLU |
| LLaMA-7B | `--model llama-7b` | 4096 | 11008 | SiLU |
| TinyLlama | `--model tinyllama` | 2048 | 5632 | SiLU |
| Custom | `--d-model N --d-ff M` | N | M | any |

To use a **real exported model** instead of synthetic weights:

```python
import torch
from transformers import GPT2Model

model = GPT2Model.from_pretrained("gpt2")
dummy = torch.zeros(1, 1, dtype=torch.long)
torch.onnx.export(model, dummy, "gpt2_real.onnx", opset_version=14)
```

Then pass `gpt2_real.onnx` to the extractor.

---

## Verified results

Measured on ESBMC 6.8.0, Z3 v4.8.9, Linux x86_64.

### Method 1 — Fixed-point Q8.8 with ReLU proxy

| Harness | Dimensions | Activation | Time | Result |
|---------|-----------|-----------|------|--------|
| `mini_ffn_fxp_test.c` | 2×4×2 | ReLU | 0.25 s | **SUCCESSFUL** |
| `gpt2_2x4.c` | 2×4×2 | ReLU proxy | 0.25 s | **SUCCESSFUL** |
| `gpt2_4x8.c` | 4×8×4 | ReLU proxy | 12.4 s | **SUCCESSFUL** |
| `llama_4x8.c` | 4×8×4 | ReLU proxy | 8.6 s | **SUCCESSFUL** |
| `gpt2_4x16.c` | 4×16×4 | ReLU proxy | timeout | needs Frama-C |
| `gpt2_layer0.c` | 8×32×8 | GeLU LUT | timeout | needs Frama-C |

### Method 2 — QNNVerifier-style abstract activation (exact GeLU/SiLU)

Generated by `ffn_qnn_generator.py` using the ACASXU_Reluplex_fxp technique:
per-neuron `__ESBMC_assume` interval injection replaces the activation function.

**Optimizations discovered (vs original QNNVerifier encoding):**
- Use `int32_t` for hidden neurons (values bounded to small ranges << 2^31)
- Skip dead Layer-1 pre-activation code in abstract mode (inputs decoupled)
- Use pure 32-bit arithmetic in `fxp32_mult` (products bounded << INT32_MAX)
- Use Boolector instead of Z3 (much faster for this BV structure)

Result: **18–27× speedup** over the initial Z3+int64+dead-code encoding.

| Harness | Dims | Model | Activation | Solver | Time | Result |
|---------|------|-------|-----------|--------|------|--------|
| `gpt2_2x4_qnn.c` | 2×4×2 | GPT-2 | GeLU (exact) | Boolector | 0.30 s | **SUCCESSFUL** |
| `gpt2_4x8_qnn.c` | 4×8×4 | GPT-2 | GeLU (exact) | Boolector | 5.3 s | **SUCCESSFUL** |
| `llama-7b_4x8_qnn.c` | 4×8×4 | LLaMA-7B | SiLU (exact) | Boolector | 4.3 s | **SUCCESSFUL** |
| `gpt2_4x16_qnn.c` | 4×16×4 | GPT-2 | GeLU (exact) | Boolector | 27 s | **SUCCESSFUL** |

Previous (unoptimized, Z3 + int64 + dead pre-act code): 4×8 took 94–115 s, 4×16 timed out (>20 min).

**Trade-off:** Method 2 proves exact activation semantics (no ReLU proxy approximation)
at the cost of one nondet symbolic variable per hidden neuron, making the solver's task harder.
Method 1 is faster but its ReLU proxy introduces a soundness approximation for GeLU/SiLU.

Verify any harness with:
```bash
esbmc <file>.c --no-unwinding-assertions --boolector
```

---

## Scalability and next steps

### Why fixed-point is faster than float

`--floatbv` uses the full IEEE 754 SMT theory — exponentially harder for solvers. Fixed-point reduces to **bit-vector arithmetic**, which Z3/Bitwuzla solve efficiently.

### Complexity boundary

| Method | d_ff bound | Solver | Time |
|--------|-----------|--------|------|
| Method 1 (ReLU proxy) | ≤ 8 | Z3 | seconds |
| Method 1 (ReLU proxy) | > 8 | Z3 | timeout |
| Method 2 (QNN abstract, optimized) | ≤ 8 | Boolector | < 6 s |
| Method 2 (QNN abstract, optimized) | 16 | Boolector | ~27 s |
| Method 2 (QNN abstract, unoptimized) | ≤ 8 | Z3 | 94–115 s |
| Method 2 (QNN abstract, unoptimized) | 16 | Z3 | timeout (>20 min) |

The QNN method introduces one nondet variable per hidden neuron, so Z3 complexity
scales with d_ff. Tighter analytical bounds (via Frama-C EVA) would reduce the
interval width and speed up verification.

### To verify larger slices: Frama-C interval tightening

The `ffn_verifier.py` orchestrator has `--use-framac` support. When Frama-C is installed:

```bash
python ffn_verifier.py gpt2_ffn.onnx \
    --d-model-max 8 --d-ff-max 32 \
    --use-framac --solver z3
```

Frama-C EVA computes concrete bounds for intermediate variables, which are injected as `__ESBMC_assume` statements — dramatically reducing the SMT search space (the core technique of QNNVerifier). `ffn_qnn_generator.py` already uses the same injection pattern; with Frama-C bounds the intervals would be tighter, enabling larger d_ff.

### To verify a real ONNX model (e.g., HuggingFace export)

```python
from onnx_ffn_extractor import extract_ffn
from ffn_fxp_generator import generate_fxp_c

layer = extract_ffn("my_model.onnx", layer_idx=0, d_model_max=4, d_ff_max=8)
generate_fxp_c(layer, "my_model_fxp.c")
# Then: esbmc my_model_fxp.c --overflow-check --unwind 9 --z3
```

---

## Dependencies

```
numpy       # weight manipulation and quantisation
onnx        # model loading and ONNX graph traversal
esbmc       # model checker (bundled at QNNVerifier/esbmc-6.8.0/esbmc)
frama-c     # optional — interval tightening for larger slices
```

Install Python deps:
```bash
pip install numpy onnx
```
