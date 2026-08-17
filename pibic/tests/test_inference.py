"""
Caso 2 — segurança de kernels de inferência.

NOTA SOBRE A VERSÃO ANTERIOR DESTE ARQUIVO. Ele declarava 15 parâmetros
("GEMM naive", "GEMM tiled", …) mas o argumento `kernel_task` **nunca era usado
no corpo** — só interpolado na mensagem de erro. Os 15 casos executavam o
comando idêntico sobre o arquivo idêntico: era 1 teste apresentado como 15, para
bater a meta de ">10 tasks" do PRD. Contagem inflada de teste é o mesmo defeito
que o board acusa no artigo — número sem lastro —, então foi desfeita aqui.

Dois defeitos reais que a contagem inflada escondia, ambos corrigidos:

  1. O arquivo se chamava `kernels_benchmarks.cpp`, mas não contém um único
     recurso de C++. O frontend C++ do ESBMC 6.8.0 abortava com CONVERSION
     ERROR, então **nenhum dos 15 testes verificava coisa alguma**. Renomeado
     para `.c`, o mesmo conteúdo verifica normalmente.

  2. Com `--unwind 4` e sem `--no-unwinding-assertions`, a violação de
     unwinding aparece ANTES do vazamento e o mascara. Medido:

         unwind=4                          -> ['assertion failure', 'loop unwinding failure']
         unwind=4, no_unwinding_assertions -> ['memory leak', 'assertion failure']
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_verify.esbmc_caller import UNSAFE, run_esbmc  # noqa: E402
from core_verify.SMT_feedback_parser import FeedbackTrace  # noqa: E402

KERNEL_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "cases", "inference_safety",
    "kernels_benchmarks.c"))

FLAGS = dict(memory_leak_check=True, overflow_check=True, no_pointer_check=True,
             no_unwinding_assertions=True, unwind=4, z3=True)


def test_kernel_compila_e_produz_veredito():
    """Guarda contra a regressão do `.cpp`: o arquivo tem de CHEGAR a um
    veredito. Antes ele abortava em CONVERSION ERROR e o teste não percebia."""
    r = run_esbmc(KERNEL_FILE, timeout=180, **FLAGS)
    assert r.verificou, (
        f"ESBMC não emitiu veredito: status={r.status} rc={r.returncode}\n"
        f"{r.output[-800:]}"
    )


def test_vazamento_injetado_e_detectado():
    """`verify_all_bounds()` faz três malloc e nenhum free — de propósito."""
    r = run_esbmc(KERNEL_FILE, timeout=180, **FLAGS)
    assert r.status == UNSAFE, f"esperado UNSAFE, obtido {r.status}"
    violacoes = FeedbackTrace(r.output).violations
    assert "memory leak" in violacoes, (
        f"vazamento injetado não detectado; violações: {violacoes}"
    )
