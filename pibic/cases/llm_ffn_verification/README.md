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
        └─── ffn_fxp_generator.py   # Fixed-point Q8.8 harness  ← recommended
                   └─ mini_ffn_fxp_test.c (smoke test)
                        │
                        ▼
              ffn_verifier.py       # Orchestrator: extract → generate → ESBMC
                        │
                        ▼
              esbmc_caller.py       # Subprocess wrapper (../../../core_verify/)
```

---

## Files

| File | Purpose |
|------|---------|
| `gpt2_ffn_builder.py` | Build GPT-2 / LLaMA / TinyLlama FFN as ONNX (no torch) |
| `onnx_ffn_extractor.py` | Extract FFN weights from any ONNX transformer model |
| `ffn_c_generator.py` | Generate float C harness (good for tiny dims) |
| `ffn_fxp_generator.py` | Generate fixed-point Q8.8 C harness **(recommended)** |
| `ffn_verifier.py` | End-to-end orchestrator CLI |
| `gelu_lut_gen.py` | Generate GeLU LUT header (1000 entries, [-5, 5]) |
| `silu_lut_gen.py` | Generate SiLU LUT header (1600 entries, [-8, 8]) |
| `gelu_lut.h` | Pre-generated GeLU lookup table |
| `silu_lut.h` | Pre-generated SiLU lookup table |
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

### 2. Verify GPT-2 layer 0 (4×8 slice, ~12 seconds)

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

### 3. Verify LLaMA-7B layer 0 (4×8 slice, ~9 seconds)

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

| Harness | Dimensions | Arithmetic | Activation | Time | Result |
|---------|-----------|-----------|-----------|------|--------|
| `mini_ffn_fxp_test.c` | 2×4×2 | Fixed-point Q8.8 | ReLU | 0.25 s | **SUCCESSFUL** |
| `gpt2_2x4.c` | 2×4×2 | Fixed-point Q8.8 | ReLU proxy | 0.25 s | **SUCCESSFUL** |
| `gpt2_4x8.c` | 4×8×4 | Fixed-point Q8.8 | ReLU proxy | 12.4 s | **SUCCESSFUL** |
| `llama_4x8.c` | 4×8×4 | Fixed-point Q8.8 | ReLU proxy | 8.6 s | **SUCCESSFUL** |
| `gpt2_4x16.c` | 4×16×4 | Fixed-point Q8.8 | ReLU proxy | timeout | needs Frama-C |
| `gpt2_layer0.c` | 8×32×8 | Float IEEE-754 | GeLU LUT | timeout | needs Frama-C |

---

## Scalability and next steps

### Why fixed-point is faster than float

`--floatbv` uses the full IEEE 754 SMT theory — exponentially harder for solvers. Fixed-point reduces to **bit-vector arithmetic**, which Z3/Bitwuzla solve efficiently.

### Complexity boundary

Without interval tightening:

```
d_model × d_ff ≤ 32  →  tractable (seconds)
d_model × d_ff > 32  →  requires Frama-C EVA interval injection
```

### To verify larger slices: Frama-C interval tightening

The `ffn_verifier.py` orchestrator has `--use-framac` support. When Frama-C is installed:

```bash
python ffn_verifier.py gpt2_ffn.onnx \
    --d-model-max 8 --d-ff-max 32 \
    --use-framac --solver z3
```

Frama-C EVA computes concrete bounds for intermediate variables, which are injected as `__ESBMC_assume` statements — dramatically reducing the SMT search space (the core technique of QNNVerifier).

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
