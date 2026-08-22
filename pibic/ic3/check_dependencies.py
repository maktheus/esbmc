#!/usr/bin/env python3
"""Diagnostica, sem sintetizar modelos, as dependencias da raia IC3/PDR.

Saida 0: pipeline AIGER pronto. Saida 2: falta Yosys/ABC.
Com ``--require-word-level``, tambem exige ``write_btor`` e ao menos um motor
BTOR2 (btormc, pono ou avr). Caminhos podem ser configurados por YOSYS e ABC.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Optional


def resolve(configured: Optional[str], candidates: list[str]) -> Optional[str]:
    if configured:
        return shutil.which(configured)
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def short_probe(argv: list[str]) -> str:
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                errors="replace", timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"falha ao consultar versao: {exc}"
    lines = (result.stdout + result.stderr).strip().splitlines()
    return lines[0] if lines else f"codigo {result.returncode}, sem versao"


def has_write_btor(yosys: Optional[str]) -> bool:
    if not yosys:
        return False
    try:
        result = subprocess.run([yosys, "-Q", "-p", "help write_btor"],
                                capture_output=True, text=True,
                                errors="replace", timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    output = (result.stdout + result.stderr).lower()
    return result.returncode == 0 and "no such command" not in output and "write_btor" in output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-word-level", action="store_true",
                        help="falha se a rota BTOR2 de KB-E07 nao estiver pronta")
    parser.add_argument("--json", action="store_true", help="emite JSON")
    args = parser.parse_args()

    yosys = resolve(os.environ.get("YOSYS"), ["yosys"])
    abc = resolve(os.environ.get("ABC"), ["yosys-abc", "abc"])
    engines = {name: shutil.which(name) for name in ("btormc", "pono", "avr")}
    btor_backend = has_write_btor(yosys)
    report = {
        "aiger_pipeline_ready": bool(yosys and abc),
        "word_level_pipeline_ready": bool(btor_backend and any(engines.values())),
        "yosys": yosys,
        "abc": abc,
        "yosys_write_btor": btor_backend,
        "word_level_engines": engines,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Dependencias IC3/PDR")
        print(f"  Yosys: {yosys or 'AUSENTE'}")
        if yosys:
            print(f"    {short_probe([yosys, '-V'])}")
        print(f"  ABC: {abc or 'AUSENTE'}")
        if abc:
            print(f"    {short_probe([abc, '-c', 'version'])}")
        print(f"  backend Yosys write_btor: {'sim' if btor_backend else 'nao'}")
        for name, path in engines.items():
            print(f"  motor {name}: {path or 'AUSENTE'}")
        print(f"  rota AIGER: {'PRONTA' if report['aiger_pipeline_ready'] else 'BLOQUEADA'}")
        print("  rota BTOR2 word-level: "
              f"{'PRONTA' if report['word_level_pipeline_ready'] else 'BLOQUEADA'}")

    required_ready = (report["word_level_pipeline_ready"] if args.require_word_level
                      else report["aiger_pipeline_ready"])
    return 0 if required_ready else 2


if __name__ == "__main__":
    sys.exit(main())
