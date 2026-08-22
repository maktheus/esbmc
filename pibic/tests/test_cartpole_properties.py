"""Ground truth das propriedades CartPole que antes eram vacuas."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "cartpole")))

from core_verify.esbmc_caller import SAFE, UNSAFE, run_esbmc  # noqa: E402
from verify_ddpg_closed_loop import (  # noqa: E402
    harness_prop_b, harness_prop_c, harness_prop_d,
)
from verify_closed_loop import (  # noqa: E402
    harness_prop_b as harness_dqn_prop_b,
    run_esbmc as run_dqn_esbmc,
)


def _run(tmp_path, name, src):
    path = tmp_path / name
    path.write_text(src, encoding="utf-8")
    return run_esbmc(str(path), z3=True,
                     no_unwinding_assertions=True, timeout=60)


def test_prop_b_dois_passos_depende_da_forca_do_controlador(tmp_path):
    """Mesmo estado: actor inerte viola; força restauradora preserva o bound.

    Em th=53 e thd=50, a força só alcança theta por thd_new no segundo passo.
    Este par falharia se o assert voltasse a observar apenas th_new, que era
    independente da rede na versão vacuamente apresentada como propriedade do
    controlador.
    """
    estado = """
    __ESBMC_assume(x == 0);
    __ESBMC_assume(xd == 0);
    __ESBMC_assume(th == 53);
    __ESBMC_assume(thd == 50);
    """
    inerte = _run(tmp_path, "prop_b_inerte.c",
                  harness_prop_b(estado + "\n    int z = 0;"))
    restaurador = _run(tmp_path, "prop_b_restaurador.c",
                       harness_prop_b(estado + "\n    int z = 900;"))

    assert inerte.status == UNSAFE, inerte.output[-800:]
    assert restaurador.status == SAFE, restaurador.output[-800:]


def test_prop_c_e_sanidade_mas_prop_d_discrimina_actor_inerte(tmp_path):
    """C vale por construção do tanh; D detecta que z=0 é controle inerte."""
    ctrl_inerte = "    int z = 0;"
    c = _run(tmp_path, "prop_c_inerte.c", harness_prop_c(ctrl_inerte))
    d = _run(tmp_path, "prop_d_inerte.c", harness_prop_d(ctrl_inerte))

    assert c.status == SAFE, c.output[-800:]
    assert d.status == UNSAFE, d.output[-800:]


def test_dqn_prop_b_dois_passos_depende_da_acao(tmp_path):
    """O assert DQN também precisa observar o passo em que F_Q alcança theta."""
    estado = """
    __ESBMC_assume(x == 0);
    __ESBMC_assume(xd == 0);
    __ESBMC_assume(th == 53);
    __ESBMC_assume(thd == 50);
    """
    forca_errada = _run(
        tmp_path, "dqn_prop_b_errada.c",
        harness_dqn_prop_b(estado + "\n    int action = 0;"),
    )
    forca_restauradora = _run(
        tmp_path, "dqn_prop_b_restauradora.c",
        harness_dqn_prop_b(estado + "\n    int action = 1;"),
    )

    assert forca_errada.status == UNSAFE, forca_errada.output[-800:]
    assert forca_restauradora.status == SAFE, forca_restauradora.output[-800:]


def test_runner_cartpole_nao_confunde_contraexemplo_com_erro(tmp_path):
    unsafe_src = """
    int main(void) {
        __ESBMC_assert(0, "falha intencional");
        return 0;
    }
    """
    path = tmp_path / "runner_unsafe.c"
    path.write_text(unsafe_src, encoding="utf-8")
    ok, _, output = run_dqn_esbmc(str(path), timeout=30)
    assert ok is False
    assert "VERIFICATION FAILED" in output

    missing = tmp_path / "nao_existe.c"
    ok, reason, output = run_dqn_esbmc(str(missing), timeout=30)
    assert ok is None
    assert reason == "PARSE_ERROR"
    assert "VERIFICATION FAILED" not in output
