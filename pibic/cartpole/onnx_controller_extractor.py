"""
onnx_controller_extractor.py — Extrai pesos do controlador DQN Cart-Pole (ONNX).

Arquitetura suportada: 4 → 24 → 24 → N  (N=2 modelo antigo; N=5 modelo novo)
Grafo ONNX: Input → Gemm → Relu → Gemm → Relu → Gemm → Output
             (camada 1)           (camada 2)           (saída)

Retorna:
    {
      "w1":    [[...]]  shape [24, 4]    pesos camada 1
      "b1":    [...]    shape [24]       viés camada 1
      "w2":    [[...]]  shape [24, 24]   pesos camada 2
      "b2":    [...]    shape [24]       viés camada 2
      "w_out": [[...]]  shape [N, 24]    pesos saída (N = número de ações)
      "b_out": [...]    shape [N]        viés saída
    }
"""

import numpy as np
import onnx


def extract_controller_weights(path):
    model   = onnx.load(path)
    tensors = {
        init.name: np.frombuffer(
            init.raw_data, dtype=np.float32
        ).reshape(init.dims)
        for init in model.graph.initializer
    }
    gemm = [n for n in model.graph.node if n.op_type == "Gemm"]
    assert len(gemm) == 3, f"Esperado 3 nós Gemm, encontrado {len(gemm)}"

    return {
        "w1":    tensors[gemm[0].input[1]].tolist(),
        "b1":    tensors[gemm[0].input[2]].tolist(),
        "w2":    tensors[gemm[1].input[1]].tolist(),
        "b2":    tensors[gemm[1].input[2]].tolist(),
        "w_out": tensors[gemm[2].input[1]].tolist(),
        "b_out": tensors[gemm[2].input[2]].tolist(),
    }


if __name__ == "__main__":
    import sys, os
    path = sys.argv[1] if len(sys.argv) > 1 else \
           os.path.join(os.path.dirname(__file__), "dqn_cartpole.onnx")
    w = extract_controller_weights(path)
    print(f"w1    : {len(w['w1'])}×{len(w['w1'][0])}")
    print(f"b1    : {len(w['b1'])} valores")
    print(f"w2    : {len(w['w2'])}×{len(w['w2'][0])}")
    print(f"b2    : {len(w['b2'])} valores")
    print(f"w_out : {len(w['w_out'])}×{len(w['w_out'][0])}")
    print(f"b_out : {w['b_out']}")
