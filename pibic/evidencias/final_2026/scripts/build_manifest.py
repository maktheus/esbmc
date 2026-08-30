"""Gera o manifesto final da rodada de auditoria."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVID = ROOT / "evidencias" / "final_2026"
CART = ROOT / "cartpole"
OUT = EVID / "MANIFESTO.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    esbmc = CART / ".." / "QNNVerifier" / "esbmc-6.8.0" / "esbmc.exe"
    try:
        version = subprocess.run([str(esbmc), "--version"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError as exc:
        version = f"unavailable: {exc}"
    inputs = [
        CART / "ddpg_actor_best.pth",
        CART / "ddpg_actor.pth",
        CART / "webapp" / "public" / "ddpg_weights.json",
        CART / "webapp" / "public" / "ddpg_weights_q88.json",
        CART / "quantization_report.json",
        CART / "ddpg_dead_neuron_results.json",
        CART / "ddpg_saturation_results.json",
        CART / "ddpg_closed_loop_results.json",
        CART / "training_history.json",
        CART / "texto_apresentacao_pibic.md",
        CART / "verify_ddpg_dead_neurons.py",
        CART / "verify_ddpg_saturation.py",
    ]
    generated = [p for p in EVID.rglob("*") if p.is_file() and p.name != OUT.name]
    manifest = {
        "round_id": "final_2026_canonical_r1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "rodada isolada de reproducao com auditoria estrutural corrigida; versoes legadas preservadas",
        "root": str(ROOT),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "esbmc": version,
            "solver_flags": ["--no-unwinding-assertions", "--boolector"],
        },
        "commands": [
            "python evidencias/final_2026/scripts/canonical_quantization.py",
            "python evidencias/final_2026/scripts/canonical_concrete_checks.py",
            "python evidencias/final_2026/scripts/build_manifest.py",
        ],
        "inputs_sha256": {str(p): sha256(p) for p in inputs if p.exists()},
        "generated_sha256": {str(p): sha256(p) for p in sorted(generated) if p.exists()},
        "results": {
            "quantization": str(EVID / "outputs" / "quantization_canonical.json"),
            "quantization_extremes": str(EVID / "outputs" / "quantization_extremes.csv"),
            "concrete_replays": str(EVID / "outputs" / "concrete_checks.json"),
        },
        "status": "partial_round_completed",
        "known_limits": [
            "A rodada usa os JSONs Float32/Q8.8 exportados; a leitura direta do checkpoint PyTorch depende de PyTorch disponível.",
            "As reproducoes concretas validam estados fixos; nao sao provas universais.",
            "A auditoria estrutural corrigida usa Z3; TIMEOUT e UNKNOWN permanecem inconclusivos.",
            "A propriedade A-direita universal permaneceu em TIMEOUT; nao ha prova de malha fechada global.",
        ],
    }
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
