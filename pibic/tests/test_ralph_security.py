"""Regressões de segurança do loop agente -> verificador."""

import json
from pathlib import Path


PIBIC = Path(__file__).resolve().parents[1]


def test_verificador_nao_executa_string_do_prd_com_eval():
    script = (PIBIC / "ralph_loop_esbmc.sh").read_text(encoding="utf-8")
    prd = json.loads((PIBIC / "ralph_prd.json").read_text(encoding="utf-8"))

    assert 'eval "$VERIFY_CMD"' not in script
    assert '"${VERIFY_ARGV[@]}"' in script
    for task in prd["tasks"]:
        assert "verify_cmd" not in task
        assert task["verify_argv"][0] == "esbmc"
        assert all(isinstance(arg, str) for arg in task["verify_argv"])


def test_loop_e_compativel_com_bash_3_e_amarra_o_alvo_canonico():
    script = (PIBIC / "ralph_loop_esbmc.sh").read_text(encoding="utf-8")

    assert "mapfile -" not in script
    assert "readarray -" not in script
    assert "os.path.realpath(os.path.abspath" in script
    assert "verify_target != target" in script


def test_veredito_exige_codigo_de_retorno_e_marcador():
    script = (PIBIC / "ralph_loop_esbmc.sh").read_text(encoding="utf-8")

    assert '[ "$VERIFY_CODE" -eq 0 ]' in script
    assert '[ "$VERIFY_CODE" -eq 1 ]' in script
    assert 'grep -q "VERIFICATION SUCCESSFUL" "$VERIFY_OUT"' in script
    assert 'grep -q "VERIFICATION FAILED" "$VERIFY_OUT"' in script
    assert '! grep -q "VERIFICATION FAILED" "$VERIFY_OUT"' in script
    assert '! grep -q "VERIFICATION SUCCESSFUL" "$VERIFY_OUT"' in script
