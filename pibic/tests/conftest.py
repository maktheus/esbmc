"""Configuração comum da suíte.

Se o binário do ESBMC não existe, a suíte **pula** com motivo explícito em vez
de acumular erros — "skipped: binário ausente" é informação; 21 tracebacks de
FileNotFoundError não são.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_verify.esbmc_caller import ESBMC_BIN  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def exige_esbmc():
    if ESBMC_BIN is None:
        pytest.skip("binário do ESBMC não encontrado — defina $ESBMC_BIN",
                    allow_module_level=True)
    return ESBMC_BIN


def pytest_report_header(config):
    return f"ESBMC: {ESBMC_BIN or 'NÃO ENCONTRADO'}"
