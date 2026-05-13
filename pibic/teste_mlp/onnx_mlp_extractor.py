"""
onnx_mlp_extractor.py — Extrai pesos de mlp_model.onnx e retorna dicts prontos para quantização.

Uso standalone:
    python onnx_mlp_extractor.py [mlp_model.onnx]

Uso como módulo:
    from onnx_mlp_extractor import extract_mlp_weights
    weights = extract_mlp_weights("mlp_model.onnx")
    # weights["w_hidden"]  → list[list[float]]  shape [4][2]
    # weights["b_hidden"]  → list[float]        shape [4]
    # weights["w_out"]     → list[float]        shape [4]
    # weights["b_out"]     → float
"""

import sys
import onnx
import numpy as np


def extract_mlp_weights(onnx_path: str) -> dict:
    """
    Lê mlp_model.onnx e devolve os pesos do MLP 2→4→1 em estruturas Python nativas.
    Espera dois nós Gemm: hidden (Gemm+ReLU) e output (Gemm+Sigmoid).
    """
    model = onnx.load(onnx_path)

    # Monta dicionário nome → ndarray
    tensors = {}
    for init in model.graph.initializer:
        arr = np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
        tensors[init.name] = arr

    # Localiza os dois nós Gemm em ordem
    gemm_nodes = [n for n in model.graph.node if n.op_type == "Gemm"]
    if len(gemm_nodes) < 2:
        raise ValueError(f"Esperava 2 nós Gemm, encontrei {len(gemm_nodes)}")

    hidden_node = gemm_nodes[0]   # inputs: [x, W_hidden, b_hidden]
    output_node  = gemm_nodes[1]  # inputs: [h, W_out,    b_out]

    W_hidden = tensors[hidden_node.input[1]]  # shape [4, 2]
    b_hidden = tensors[hidden_node.input[2]]  # shape [4]
    W_out    = tensors[output_node.input[1]]  # shape [1, 4]
    b_out    = tensors[output_node.input[2]]  # shape [1]

    return {
        "w_hidden": W_hidden.tolist(),           # [[w00,w01],[w10,w11],...]
        "b_hidden": b_hidden.tolist(),           # [b0,b1,b2,b3]
        "w_out":    W_out[0].tolist(),           # [w0,w1,w2,w3]  (flatten linha 0)
        "b_out":    float(b_out[0]),
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "mlp_model.onnx"
    w = extract_mlp_weights(path)
    print(f"Extraído de: {path}")
    for i, row in enumerate(w["w_hidden"]):
        print(f"  w_hidden[{i}] = {row}")
    print(f"  b_hidden   = {w['b_hidden']}")
    print(f"  w_out      = {w['w_out']}")
    print(f"  b_out      = {w['b_out']}")
