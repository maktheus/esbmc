"""
Caso 4 — resiliência do PID sob perfis de caos.

O QUE MUDOU E POR QUÊ. Antes, os cinco perfis passavam em milissegundos com
`assert result is not None` — asserção logicamente impossível de falhar, já que
`run_esbmc` ou devolve um objeto ou levanta exceção. Eram rápidos por um motivo
pior que o teste fraco: **o ruído injetado nunca chegava ao sistema**. Cada
perfil declarava uma variável própria (`noise_uniform`, `noise_drift`, …) que
ficava órfã, enquanto o template somava um `float noise = 0.0f;`. Os cinco
perfis verificavam o mesmo PID sem ruído nenhum.

Com o ruído realmente ligado (ver `chaos_generator.py`), a dificuldade real
aparece — e é o resultado interessante do Caso 4:

    perfil        veredito    tempo      por quê
    Impulse       UNSAFE       27 s      ruído em 3 valores discretos (0, ±100)
    Uniform       TIMEOUT    >240 s      faixa float contínua sob --floatbv
    Drift         TIMEOUT    >240 s      idem

Ou seja: **a decidibilidade depende da cardinalidade do ruído**, não do PID.
Perfis discretos decidem rápido; faixas contínuas não decidem em minutos com
codificação de float exata.

Por isso o teste NÃO exige veredito de todos os perfis. Ele exige que o harness
seja bem-formado — que o ESBMC compile e execute — e registra o que o solver
conseguiu decidir. Exigir veredito seria transformar um limite conhecido do
solver em falha de teste; aceitar qualquer coisa seria voltar à tautologia.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cases.control_chaos_testing.chaos_generator import (  # noqa: E402
    PERFIS, inject_pid_chaos,
)
from core_verify.esbmc_caller import (  # noqa: E402
    PARSE_ERROR, TIMEOUT, UNSAFE, USAGE_ERROR, run_esbmc,
)

TEMPLATE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "cases", "control_chaos_testing",
    "pid_template.c"))

#: Perfis com ruído de cardinalidade pequena decidem; faixas contínuas não.
DECIDEM = {"Impulse"}


@pytest.mark.parametrize("noise_type", sorted(PERFIS))
def test_harness_de_caos_e_bem_formado(noise_type, tmp_path):
    """O C gerado tem de compilar e o motor tem de rodar.

    PARSE_ERROR pegaria literais inválidos — foi assim que `100f` do perfil
    Drift apareceu. USAGE_ERROR pegaria flag inexistente. Ambos significam que
    NADA foi verificado, e é isso que este teste protege.
    """
    out = str(tmp_path / f"pid_{noise_type}.c")   # nunca na árvore de fontes
    inject_pid_chaos(TEMPLATE, out, noise_type)

    src = open(out).read()
    assert "float noise " in src, "a injeção tem de declarar `noise`"
    assert "0.0f + noise" in src, "o ruído tem de alcançar measured_sensor"

    r = run_esbmc(out, z3=True, floatbv=True, unwind=6,
                  no_pointer_check=True, timeout=90)

    assert r.status not in (PARSE_ERROR, USAGE_ERROR), (
        f"harness mal-formado para {noise_type}: status={r.status} "
        f"rc={r.returncode}\n{r.output[-800:]}"
    )
    print(f"{noise_type}: {r.status} em {r.time_taken:.1f}s")


@pytest.mark.parametrize("noise_type", sorted(DECIDEM))
def test_perfil_discreto_encontra_violacao(noise_type, tmp_path):
    """Impulse injeta ±100 num setpoint de 10 — o PID tem de estourar os
    limites. Este é o teste com conteúdo: exige veredito, não só execução."""
    out = str(tmp_path / f"pid_{noise_type}.c")
    inject_pid_chaos(TEMPLATE, out, noise_type)
    r = run_esbmc(out, z3=True, floatbv=True, unwind=6,
                  no_pointer_check=True, timeout=180)
    assert r.status == UNSAFE, (
        f"esperado contraexemplo para {noise_type}; obtido {r.status}"
    )


@pytest.mark.parametrize("noise_type", sorted(set(PERFIS) - DECIDEM))
def test_perfil_continuo_nao_decide_documentado(noise_type, tmp_path):
    """Registra a limitação medida em vez de escondê-la.

    Se algum dia um destes passar a decidir, este teste falha — e isso é
    informação boa: significa que o solver ou a codificação melhoraram, e a
    tabela do Caso 4 precisa ser atualizada.
    """
    out = str(tmp_path / f"pid_{noise_type}.c")
    inject_pid_chaos(TEMPLATE, out, noise_type)
    r = run_esbmc(out, z3=True, floatbv=True, unwind=6,
                  no_pointer_check=True, timeout=90)
    assert r.status == TIMEOUT, (
        f"{noise_type} passou a decidir ({r.status}) — atualize a tabela do "
        f"Caso 4 e mova o perfil para DECIDEM"
    )
