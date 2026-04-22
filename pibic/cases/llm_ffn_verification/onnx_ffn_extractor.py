"""
Extract a single FFN (Feed-Forward Network) layer from an ONNX LLM model.

A transformer FFN has the shape:
    input  →  Linear(d_model → d_ff)  →  GeLU/ReLU  →  Linear(d_ff → d_model)  →  output

Supported ONNX patterns:
  - GPT-2 style: Gemm(transB=1) → Gelu → Gemm(transB=1)
  - Generic:     MatMul + Add   → Gelu → MatMul + Add

Usage:
    from onnx_ffn_extractor import extract_ffn, FFNLayer

    layer = extract_ffn("model.onnx", layer_idx=0, d_model_max=8, d_ff_max=32)
    print(layer)           # FFNLayer(d_model=8, d_ff=32, activation='gelu')
    print(layer.W1.shape)  # (32, 8)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FFNLayer:
    """Weights of one transformer FFN block."""
    W1: np.ndarray   # shape [d_ff, d_model]  — up-projection
    b1: np.ndarray   # shape [d_ff]
    W2: np.ndarray   # shape [d_model, d_ff]  — down-projection
    b2: np.ndarray   # shape [d_model]
    activation: str = "gelu"
    source_layer_idx: int = 0

    @property
    def d_model(self) -> int:
        return self.W1.shape[1]

    @property
    def d_ff(self) -> int:
        return self.W1.shape[0]

    def slice(self, d_model_max: int, d_ff_max: int) -> "FFNLayer":
        """
        Return a reduced FFN for tractable ESBMC verification.
        Takes the first d_model_max input dims and first d_ff_max hidden neurons.
        """
        dm = min(self.d_model, d_model_max)
        df = min(self.d_ff, d_ff_max)
        return FFNLayer(
            W1=self.W1[:df, :dm].copy(),
            b1=self.b1[:df].copy(),
            W2=self.W2[:dm, :df].copy(),
            b2=self.b2[:dm].copy(),
            activation=self.activation,
            source_layer_idx=self.source_layer_idx,
        )

    def activation_bounds(self) -> tuple[float, float]:
        """
        Conservative output range after GeLU given zero-initialised hidden state.
        Used to seed __ESBMC_assert bounds in generated C.
        """
        # GeLU output is always >= min(0, x) — safe lower bound is 0 for ReLU-like
        return (0.0, float("inf"))

    def max_output_magnitude(self, input_bound: float = 4.0) -> float:
        """
        Compute loose upper bound on |output| using matrix norms.
        |output| <= ||W2||_1 * ||W1||_1 * input_bound + ||b2||_inf + ||b1||_inf scaled
        """
        max_hidden = np.max(np.abs(self.W1), axis=1) * input_bound * self.d_model + np.abs(self.b1)
        # GeLU clamps negatives near 0, so effective max is max_hidden
        max_out = np.max(np.abs(self.W2), axis=1) * np.sum(max_hidden) + np.abs(self.b2)
        return float(np.max(max_out))

    def __repr__(self) -> str:
        return (
            f"FFNLayer(d_model={self.d_model}, d_ff={self.d_ff}, "
            f"activation='{self.activation}', layer_idx={self.source_layer_idx})"
        )


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------

def _require_onnx() -> None:
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "onnx package not found. Install with: pip install onnx"
        )


def _get_initializer(initializers: dict, name: str) -> Optional[np.ndarray]:
    return initializers.get(name)


def _gemm_weight_bias(
    node, initializers: dict, transpose_b: bool = True
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract (weight, bias) from a Gemm or MatMul+Add node.
    For Gemm: inputs are [X, W, B].
    For MatMul: only [X, W] — bias comes from a subsequent Add node.
    """
    if node.op_type == "Gemm":
        inputs = list(node.input)
        W = _get_initializer(initializers, inputs[1]) if len(inputs) > 1 else None
        b = _get_initializer(initializers, inputs[2]) if len(inputs) > 2 else None
        if W is not None:
            # Gemm uses transB=1 by default in many exporters → W is already [d_ff, d_in]
            trans_b = any(
                attr.name == "transB" and attr.i == 1 for attr in node.attribute
            )
            if trans_b:
                pass  # W shape is [d_ff, d_model] — correct for our convention
            else:
                W = W.T
        return W, b

    if node.op_type == "MatMul":
        inputs = list(node.input)
        W = _get_initializer(initializers, inputs[1]) if len(inputs) > 1 else None
        # bias must come from a subsequent Add — caller handles this
        return W, None

    return None, None


def _build_graph_maps(graph) -> tuple[dict, dict]:
    """
    Returns:
      out_to_node : output_name  → node
      inp_to_nodes: input_name   → [nodes that consume it]
    """
    out_to_node: dict = {}
    inp_to_nodes: dict = {}
    for node in graph.node:
        for out in node.output:
            out_to_node[out] = node
        for inp in node.input:
            inp_to_nodes.setdefault(inp, []).append(node)
    return out_to_node, inp_to_nodes


_ACTIVATION_OPS = {
    "Gelu", "Relu", "Silu", "Swish",
    "FastGelu", "BiasGelu",                # common HuggingFace exported names
    "Tanh",                                 # some older models
    "Erf", "Add",                           # GELU via erf: 0.5*x*(1+erf(x/sqrt(2)))
}


def _follow_to_next_linear(
    start_output: str,
    out_to_node: dict,
    inp_to_nodes: dict,
    max_hops: int = 8,
) -> Optional[object]:
    """
    Follow the computation graph from start_output through activation ops
    until we hit a Gemm or MatMul node (the down-projection).
    Returns that node or None.
    """
    current = start_output
    for _ in range(max_hops):
        consumers = inp_to_nodes.get(current, [])
        if not consumers:
            return None
        # pick the first non-initializer consumer
        consumer = consumers[0]
        if consumer.op_type in ("Gemm", "MatMul"):
            return consumer
        # pass through activation / reshape / transpose ops
        if consumer.output:
            current = consumer.output[0]
        else:
            return None
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_ffn(
    onnx_path: str,
    layer_idx: int = 0,
    d_model_max: int = 8,
    d_ff_max: int = 32,
    activation: str = "gelu",
) -> FFNLayer:
    """
    Extract and slice one FFN layer from an ONNX model.

    Parameters
    ----------
    onnx_path   : path to .onnx file
    layer_idx   : which FFN layer to extract (0 = first transformer block)
    d_model_max : max input/output dimension slice for verification
    d_ff_max    : max hidden dimension slice for verification
    activation  : activation function name (informational; affects C code gen)

    Returns
    -------
    FFNLayer with W1[d_ff_max, d_model_max], b1[d_ff_max],
                  W2[d_model_max, d_ff_max], b2[d_model_max]
    """
    _require_onnx()
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    graph = model.graph

    initializers = {
        init.name: numpy_helper.to_array(init) for init in graph.initializer
    }
    out_to_node, inp_to_nodes = _build_graph_maps(graph)

    # ---- Find all (up-proj, down-proj) pairs --------------------------------
    candidate_pairs: list[tuple] = []  # list of (W1, b1, W2, b2)

    visited_nodes: set = set()

    for node in graph.node:
        if node.op_type not in ("Gemm", "MatMul"):
            continue
        if id(node) in visited_nodes:
            continue

        W1, b1 = _gemm_weight_bias(node, initializers)
        if W1 is None or W1.ndim != 2:
            continue

        # For MatMul, look for an immediately following Add for bias
        node_output = node.output[0]
        if node.op_type == "MatMul" and b1 is None:
            add_consumers = [
                n for n in inp_to_nodes.get(node_output, [])
                if n.op_type == "Add"
            ]
            if add_consumers:
                add_node = add_consumers[0]
                for inp in add_node.input:
                    candidate_b = _get_initializer(initializers, inp)
                    if candidate_b is not None and candidate_b.ndim == 1:
                        b1 = candidate_b
                        node_output = add_node.output[0]
                        break

        if b1 is None:
            b1 = np.zeros(W1.shape[0], dtype=np.float32)

        # Follow through activation to find down-projection
        down_node = _follow_to_next_linear(node_output, out_to_node, inp_to_nodes)
        if down_node is None or id(down_node) in visited_nodes:
            continue

        W2, b2 = _gemm_weight_bias(down_node, initializers)
        if W2 is None or W2.ndim != 2:
            continue

        # For MatMul down-proj, check for Add bias
        down_output = down_node.output[0]
        if down_node.op_type == "MatMul" and b2 is None:
            add_consumers = [
                n for n in inp_to_nodes.get(down_output, [])
                if n.op_type == "Add"
            ]
            if add_consumers:
                add_node = add_consumers[0]
                for inp in add_node.input:
                    candidate_b = _get_initializer(initializers, inp)
                    if candidate_b is not None and candidate_b.ndim == 1:
                        b2 = candidate_b
                        break

        if b2 is None:
            b2 = np.zeros(W2.shape[0], dtype=np.float32)

        # Validate shapes: W1[d_ff, d_model], W2[d_model, d_ff]
        d_ff_candidate = W1.shape[0]
        d_model_candidate = W1.shape[1]
        if W2.shape == (d_model_candidate, d_ff_candidate):
            visited_nodes.add(id(node))
            visited_nodes.add(id(down_node))
            candidate_pairs.append((
                W1.astype(np.float32),
                b1.astype(np.float32),
                W2.astype(np.float32),
                b2.astype(np.float32),
            ))

    if not candidate_pairs:
        raise ValueError(
            f"No FFN layers found in {onnx_path}. "
            "Ensure the model is a standard transformer exported with opset >= 11."
        )

    if layer_idx >= len(candidate_pairs):
        raise IndexError(
            f"layer_idx={layer_idx} out of range; found {len(candidate_pairs)} FFN layers."
        )

    W1, b1, W2, b2 = candidate_pairs[layer_idx]
    layer = FFNLayer(W1=W1, b1=b1, W2=W2, b2=b2,
                     activation=activation, source_layer_idx=layer_idx)

    print(f"Extracted {layer}  →  slicing to d_model={d_model_max}, d_ff={d_ff_max}")
    return layer.slice(d_model_max, d_ff_max)


def extract_ffn_from_numpy(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    activation: str = "gelu",
    d_model_max: int = 8,
    d_ff_max: int = 32,
) -> FFNLayer:
    """
    Build an FFNLayer directly from numpy arrays (for unit tests or custom models).
    Slices to verification dimensions automatically.
    """
    layer = FFNLayer(
        W1=W1.astype(np.float32),
        b1=b1.astype(np.float32),
        W2=W2.astype(np.float32),
        b2=b2.astype(np.float32),
        activation=activation,
    )
    return layer.slice(d_model_max, d_ff_max)
