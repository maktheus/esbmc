"""
Casos ground-truth pareados — `roadmap.md` §4.1, "TDD Inverso".

Estes testes existem por um motivo específico: a suíte anterior só tinha
asserções **negativas** (`assert not result.is_safe`), e `is_safe` era falso
para qualquer modo de falha — flag inválida, erro de parsing, arquivo ausente.
A suíte ficava verde sem nunca ter verificado nada, e nenhum teste era capaz de
detectar isso.

O par safe/unsafe conserta o buraco: um caso que **deve** provar e um que
**deve** falhar. Se o wrapper regredir e passar a colapsar erros em "inseguro",
o caso positivo quebra imediatamente.
"""

import textwrap

import pytest

from core_verify.esbmc_caller import (
    PARSE_ERROR, SAFE, UNSAFE, USAGE_ERROR, run_esbmc,
)


def _escreve(tmp_path, nome, src):
    p = tmp_path / nome
    p.write_text(textwrap.dedent(src))
    return str(p)


# ─── o par: mesma propriedade, vereditos opostos ─────────────────────────────

def test_caso_seguro_prova(tmp_path):
    """DEVE provar. É o teste que a suíte antiga não tinha."""
    f = _escreve(tmp_path, "seguro.c", """
        int main(void) {
            int x = 0;
            for (int i = 0; i < 5; i++) x += 2;
            __ESBMC_assert(x == 10, "x termina em 10");
            return 0;
        }
    """)
    r = run_esbmc(f, unwind=10, timeout=120)
    assert r.status == SAFE, f"esperado SAFE, obtido {r.status}\n{r.output[-800:]}"


def test_caso_inseguro_encontra_contraexemplo(tmp_path):
    """DEVE falhar, com contraexemplo — não por erro de execução."""
    f = _escreve(tmp_path, "inseguro.c", """
        int main(void) {
            int x = 0;
            for (int i = 0; i < 5; i++) x += 2;
            __ESBMC_assert(x == 11, "propriedade falsa de proposito");
            return 0;
        }
    """)
    r = run_esbmc(f, unwind=10, timeout=120)
    assert r.status == UNSAFE, f"esperado UNSAFE, obtido {r.status}"
    assert r.verificou, "UNSAFE tem de significar veredito, nao erro"


# ─── os modos de falha precisam ser distinguíveis de UNSAFE ──────────────────

def test_arquivo_inexistente_nao_vira_unsafe(tmp_path):
    r = run_esbmc(str(tmp_path / "nao_existe.c"), timeout=60)
    assert r.status == PARSE_ERROR
    assert not r.verificou
    assert not r.is_safe          # continua falso...
    assert not r.is_unsafe        # ...mas nao e "bug encontrado"


def test_fonte_quebrado_nao_vira_unsafe(tmp_path):
    f = _escreve(tmp_path, "quebrado.c", "int main(void) { isto nao e C ;")
    r = run_esbmc(f, timeout=60)
    assert r.status == PARSE_ERROR
    assert not r.verificou


def test_flag_invalida_nao_vira_unsafe(tmp_path):
    """`--bounds-check` nao existe no ESBMC: os checks de bounds sao ligados por
    PADRAO e so ha a forma negativa. A suite antiga passava essa flag em 20
    testes, entao o ESBMC abortava com rc=64 e nunca verificava nada."""
    f = _escreve(tmp_path, "trivial.c", "int main(void) { return 0; }")
    r = run_esbmc(f, timeout=60, extra_args=["--bounds-check"])
    assert r.status == USAGE_ERROR
    assert not r.verificou


def test_no_bounds_check_e_a_forma_valida(tmp_path):
    f = _escreve(tmp_path, "trivial.c", "int main(void) { return 0; }")
    r = run_esbmc(f, timeout=60, no_bounds_check=True)
    assert r.verificou, f"a forma negativa deve rodar; obtido {r.status}"


def test_timeout_nao_vira_unsafe(tmp_path):
    """Timeout e INDECISO. Le-lo como veredito e o mesmo erro que ler
    'Timeout' do PDR como 'seguro'."""
    f = _escreve(tmp_path, "pesado.c", """
        int main(void) {
            unsigned a = 0, b = 0;
            for (unsigned i = 0; i < 1000000u; i++) { a += i * 7u; b ^= a << 3; }
            __ESBMC_assert(a != b, "propriedade cara");
            return 0;
        }
    """)
    r = run_esbmc(f, unwind=200000, timeout=3)
    assert not r.verificou, f"esperado indeciso, obtido {r.status}"
    assert not r.is_unsafe


# ─── a linha de comando, sem subprocess ──────────────────────────────────────

@pytest.mark.parametrize("kw,flag", [
    ({"z3": True}, "--z3"),
    ({"floatbv": True}, "--floatbv"),
    ({"k_induction": True}, "--k-induction"),
    ({"overflow_check": True}, "--overflow-check"),
    ({"no_bounds_check": True}, "--no-bounds-check"),
    ({"no_pointer_check": True}, "--no-pointer-check"),
])
def test_flags_conhecidas_entram_no_comando(kw, flag):
    from core_verify.esbmc_caller import build_esbmc_cmd
    assert flag in build_esbmc_cmd("x.c", **kw)


@pytest.mark.parametrize("ausente", ["--bounds-check", "--pointer-check"])
def test_flags_inexistentes_nunca_sao_emitidas(ausente):
    """Rede de segurança contra a regressão original."""
    from core_verify.esbmc_caller import build_esbmc_cmd
    cmd = build_esbmc_cmd("x.c", no_bounds_check=True, no_pointer_check=True,
                          z3=True, floatbv=True, overflow_check=True)
    assert ausente not in cmd
