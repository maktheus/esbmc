"""
Build a GPT-2-scale FFN as a valid ONNX file using only numpy + onnx.
No torch or transformers required.

GPT-2 small architecture (layer 0 FFN):
    d_model = 768
    d_ff    = 3072   (= 4 * d_model)
    activation: GeLU

The ONNX graph structure generated:
    input [1, d_model]
      └─ Gemm(W1[d_ff, d_model], b1[d_ff], transB=1)  →  [1, d_ff]
           └─ Gelu                                      →  [1, d_ff]
                └─ Gemm(W2[d_model, d_ff], b2[d_model], transB=1)  →  [1, d_model]

Weights are sampled from N(0, 0.02) — same initialisation as original GPT-2.

Usage:
    python gpt2_ffn_builder.py                  # writes gpt2_ffn.onnx
    python gpt2_ffn_builder.py --model llama    # writes llama_ffn.onnx (SiLU, d_model=4096, d_ff=11008)
    python gpt2_ffn_builder.py --out custom.onnx --d-model 64 --d-ff 256
"""

from __future__ import annotations
import argparse
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

PRESETS = {
    "gpt2": dict(d_model=768,  d_ff=3072,  activation="gelu", std=0.02),
    "gpt2-medium": dict(d_model=1024, d_ff=4096,  activation="gelu", std=0.02),
    "llama-7b":  dict(d_model=4096, d_ff=11008, activation="silu", std=0.02),
    "tinyllama": dict(d_model=2048, d_ff=5632,  activation="silu", std=0.02),
    "custom":    dict(d_model=64,   d_ff=256,   activation="gelu", std=0.02),
}


# ---------------------------------------------------------------------------
# ONNX graph builder
# ---------------------------------------------------------------------------

def _make_gemm(name: str, A_name: str, W: np.ndarray, b: np.ndarray) -> tuple:
    """
    Return (node, [initializers]) for a Gemm node:
        Y = A @ W^T + b   (transB=1)
    W shape: [out_features, in_features]
    b shape: [out_features]
    """
    W_name = f"{name}_W"
    b_name = f"{name}_b"
    out_name = f"{name}_out"

    W_init = numpy_helper.from_array(W.astype(np.float32), name=W_name)
    b_init = numpy_helper.from_array(b.astype(np.float32), name=b_name)

    node = helper.make_node(
        "Gemm",
        inputs=[A_name, W_name, b_name],
        outputs=[out_name],
        name=name,
        transB=1,
        alpha=1.0,
        beta=1.0,
    )
    return node, out_name, [W_init, b_init]


def build_ffn_onnx(
    d_model: int,
    d_ff: int,
    activation: str = "gelu",
    std: float = 0.02,
    seed: int = 42,
    output_path: str = "ffn.onnx",
) -> str:
    """
    Build and save an ONNX model containing a single transformer FFN block.

    Returns the path to the saved file.
    """
    rng = np.random.default_rng(seed)

    # Sample weights with GPT-2-style initialisation
    W1 = rng.normal(0, std, (d_ff, d_model)).astype(np.float32)
    b1 = np.zeros(d_ff, dtype=np.float32)
    W2 = rng.normal(0, std / np.sqrt(2), (d_model, d_ff)).astype(np.float32)
    b2 = np.zeros(d_model, dtype=np.float32)

    # ---- Graph inputs / outputs --------------------------------------------
    graph_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, d_model]
    )
    graph_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, d_model]
    )

    # ---- Nodes -------------------------------------------------------------
    nodes = []
    initializers = []

    # Up-projection
    gemm1_node, gemm1_out, gemm1_inits = _make_gemm("ffn_up", "input", W1, b1)
    nodes.append(gemm1_node)
    initializers.extend(gemm1_inits)

    # Activation — expressed with standard opset-13 ops so onnx.checker passes
    act_out = "act_out"
    if activation.lower() == "gelu":
        # GeLU(x) = 0.5 * x * (1 + Erf(x / sqrt(2)))
        # Uses: Div, Erf, Add, Mul — all in opset 13
        sqrt2_init = numpy_helper.from_array(
            np.array([1.4142135], dtype=np.float32), name="sqrt2_const"
        )
        half_init = numpy_helper.from_array(
            np.array([0.5], dtype=np.float32), name="half_const"
        )
        one_init = numpy_helper.from_array(
            np.array([1.0], dtype=np.float32), name="one_const"
        )
        initializers.extend([sqrt2_init, half_init, one_init])

        div_out  = "gelu_div"
        erf_out  = "gelu_erf"
        add_out  = "gelu_add"
        mul1_out = "gelu_mul1"

        nodes += [
            helper.make_node("Div",  [gemm1_out, "sqrt2_const"], [div_out],  "gelu_div"),
            helper.make_node("Erf",  [div_out],                  [erf_out],  "gelu_erf"),
            helper.make_node("Add",  [erf_out,   "one_const"],   [add_out],  "gelu_add"),
            helper.make_node("Mul",  [gemm1_out,  add_out],      [mul1_out], "gelu_mul1"),
            helper.make_node("Mul",  [mul1_out,  "half_const"],  [act_out],  "gelu_mul2"),
        ]
    elif activation.lower() in ("silu", "swish"):
        # SiLU(x) = x * sigmoid(x)  — both ops in opset 13
        sig_out = "silu_sigmoid"
        nodes += [
            helper.make_node("Sigmoid", [gemm1_out],           [sig_out], "silu_sigmoid"),
            helper.make_node("Mul",     [gemm1_out, sig_out],  [act_out], "silu_mul"),
        ]
    else:
        nodes.append(helper.make_node(
            "Relu", inputs=[gemm1_out], outputs=[act_out], name="ffn_act"
        ))

    # Down-projection — feeds from act_out, output named "output"
    gemm2_node, _, gemm2_inits = _make_gemm("ffn_down", act_out, W2, b2)
    # Override output name to match graph_output
    gemm2_node = helper.make_node(
        "Gemm",
        inputs=["act_out", "ffn_down_W", "ffn_down_b"],
        outputs=["output"],
        name="ffn_down",
        transB=1,
        alpha=1.0,
        beta=1.0,
    )
    nodes.append(gemm2_node)
    initializers.extend(gemm2_inits)

    # ---- Assemble graph ----------------------------------------------------
    graph = helper.make_graph(
        nodes,
        name="transformer_ffn",
        inputs=[graph_input],
        outputs=[graph_output],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.doc_string = (
        f"Single transformer FFN block: d_model={d_model}, d_ff={d_ff}, "
        f"activation={activation}"
    )
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    print(
        f"ONNX saved → {output_path}\n"
        f"  d_model={d_model}, d_ff={d_ff}, activation={activation}\n"
        f"  W1: {W1.shape}  b1: {b1.shape}\n"
        f"  W2: {W2.shape}  b2: {b2.shape}\n"
        f"  W1 range: [{W1.min():.4f}, {W1.max():.4f}]  "
        f"  W2 range: [{W2.min():.4f}, {W2.max():.4f}]"
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build GPT-2 / LLaMA FFN as ONNX")
    p.add_argument("--model", default="gpt2",
                   choices=list(PRESETS.keys()),
                   help="Preset model architecture (default: gpt2)")
    p.add_argument("--d-model", type=int, default=None,
                   help="Override d_model (input/output dim)")
    p.add_argument("--d-ff", type=int, default=None,
                   help="Override d_ff (hidden dim)")
    p.add_argument("--activation", default=None,
                   choices=["gelu", "silu", "relu"],
                   help="Override activation function")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="Output .onnx path (default: <model>_ffn.onnx)")
    args = p.parse_args()

    cfg = PRESETS[args.model].copy()
    if args.d_model:   cfg["d_model"]    = args.d_model
    if args.d_ff:      cfg["d_ff"]       = args.d_ff
    if args.activation: cfg["activation"] = args.activation

    out = args.out or f"{args.model}_ffn.onnx"
    build_ffn_onnx(**cfg, seed=args.seed, output_path=out)
