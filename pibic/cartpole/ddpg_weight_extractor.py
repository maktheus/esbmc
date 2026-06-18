"""
ddpg_weight_extractor.py — Extrai pesos do Actor DDPG a partir do checkpoint PyTorch.

Retorna dicionário com formato unificado para verificação e quantização:
  {w1, b1, w2, b2, w_out, b_out} — todos como listas Python (float).

Uso:
    from ddpg_weight_extractor import extract_ddpg_weights
    weights = extract_ddpg_weights("ddpg_actor_best.pth")
"""

import os
import torch


def extract_ddpg_weights(pth_path: str) -> dict:
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)

    return {
        "w1":    sd["net.0.weight"].tolist(),   # [24, 4]
        "b1":    sd["net.0.bias"].tolist(),      # [24]
        "w2":    sd["net.2.weight"].tolist(),   # [24, 24]
        "b2":    sd["net.2.bias"].tolist(),      # [24]
        "w_out": sd["net.4.weight"].tolist(),   # [1, 24]
        "b_out": sd["net.4.bias"].tolist(),      # [1]
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    pth = os.path.join(here, "ddpg_actor_best.pth")
    w = extract_ddpg_weights(pth)
    for k, v in w.items():
        shape = f"{len(v)}x{len(v[0])}" if isinstance(v[0], list) else f"{len(v)}"
        print(f"  {k:8s}  shape={shape}")
