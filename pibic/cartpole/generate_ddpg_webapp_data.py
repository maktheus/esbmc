"""
generate_ddpg_webapp_data.py — Agrega todos os resultados de verificacao ESBMC
em um unico JSON para o webapp.

Lê:
  - ddpg_dead_neuron_results.json
  - ddpg_saturation_results.json
  - ddpg_closed_loop_results.json
  - ddpg_weights_q88.json (para biases dos neuronios)

Gera:
  - webapp/public/ddpg_verification_data.json

Uso:
    python generate_ddpg_webapp_data.py
"""

import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEBAPP_PUBLIC = os.path.join(HERE, "webapp", "public")
SCALE = 256

sys.path.insert(0, HERE)
from ddpg_weight_extractor import extract_ddpg_weights


def load_json(name):
    path = os.path.join(HERE, name)
    with open(path) as f:
        return json.load(f)


def main():
    dead = load_json("ddpg_dead_neuron_results.json")
    sat = load_json("ddpg_saturation_results.json")
    cl = load_json("ddpg_closed_loop_results.json")

    weights_q88 = load_json(os.path.join("webapp", "public", "ddpg_weights_q88.json"))
    b1 = weights_q88["b1"]
    b2 = weights_q88["b2"]

    l1_neurons = []
    for i in range(dead["layer_1"]["total"]):
        l1_neurons.append({
            "id": i,
            "bias_q88": b1[i],
            "status": "MORTO" if i in dead["layer_1"]["dead"] else "VIVO",
        })

    l2_neurons = []
    for i in range(dead["layer_2"]["total"]):
        l2_neurons.append({
            "id": i,
            "bias_q88": b2[i],
            "status": "MORTO" if i in dead["layer_2"]["dead"] else "VIVO",
        })

    counterexamples = []

    if cl.get("property_a_right", {}).get("result") == "FAILED":
        ce = cl["property_a_right"].get("counterexample", "")
        counterexamples.append({
            "property": "property_a_right",
            "description": "theta > 5.6 graus e theta_dot >= 0, mas F <= 0",
            "state_str": ce,
            "expected_behavior": "Forca positiva (empurrar para direita)",
        })

    if cl.get("property_a_left", {}).get("result") == "FAILED":
        ce = cl["property_a_left"].get("counterexample", "")
        counterexamples.append({
            "property": "property_a_left",
            "description": "theta < -5.6 graus e theta_dot <= 0, mas F >= 0",
            "state_str": ce,
            "expected_behavior": "Forca negativa (empurrar para esquerda)",
        })

    if cl.get("property_b_safety", {}).get("result") == "FAILED":
        ce = cl["property_b_safety"].get("counterexample", "")
        counterexamples.append({
            "property": "property_b_safety",
            "description": "Estado seguro que sai da regiao segura em 1 passo",
            "state_str": ce,
            "expected_behavior": "theta permanece em [-12, 12] graus apos 1 passo",
        })

    data = {
        "model_info": {
            "architecture": "4 -> 24(ReLU) -> 24(ReLU) -> 1(Tanh) x 10",
            "controller_type": "DDPG Continuo",
            "quantization": "Q8.8 (scale=256)",
            "training_episodes": 500,
            "final_avg_score": 475.0,
        },
        "verification": {
            "dead_neurons_l1": {
                "total": dead["layer_1"]["total"],
                "dead": dead["layer_1"]["dead"],
                "neurons": l1_neurons,
            },
            "dead_neurons_l2": {
                "total": dead["layer_2"]["total"],
                "dead": dead["layer_2"]["dead"],
                "neurons": l2_neurons,
            },
            "saturation": {
                "saturated_neurons": sat.get("layer1_saturated", []),
                "output_status": sat.get("output_status", "UNKNOWN"),
            },
        },
        "closed_loop_verification": cl,
        "counterexamples": counterexamples,
    }

    os.makedirs(WEBAPP_PUBLIC, exist_ok=True)
    out_path = os.path.join(WEBAPP_PUBLIC, "ddpg_verification_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Dados gerados: {out_path}")
    print(f"  Camada 1: {dead['layer_1']['total']} neuronios, {len(dead['layer_1']['dead'])} mortos")
    print(f"  Camada 2: {dead['layer_2']['total']} neuronios, {len(dead['layer_2']['dead'])} mortos")
    print(f"  Saturados: {len(sat.get('layer1_saturated', []))}")
    print(f"  Saida: {sat.get('output_status', 'UNKNOWN')}")
    print(f"  Propriedades malha fechada: {len(cl)} verificadas")
    print(f"  Contraexemplos: {len(counterexamples)}")


if __name__ == "__main__":
    main()
