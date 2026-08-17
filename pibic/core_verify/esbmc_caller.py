"""
esbmc_caller.py — Invocação do ESBMC com veredito interpretável.

O ponto central: **um erro de execução não pode ser confundido com "bug
encontrado"**. A versão anterior derivava `is_safe` apenas da presença da string
"VERIFICATION SUCCESSFUL", então flag inválida, erro de parsing e arquivo
inexistente colapsavam todos em `is_safe=False` — exatamente o valor que
significa "a propriedade foi violada". Como toda a suíte assertava
`assert not result.is_safe`, ela ficava verde sem nunca ter verificado nada.

Códigos de retorno do ESBMC, medidos no binário 6.8.0 deste repositório:

    rc=0    VERIFICATION SUCCESSFUL
    rc=1    VERIFICATION FAILED
    rc=6    falha de front-end (erro de parsing OU arquivo inexistente)
    rc=64   flag não reconhecida

O status é derivado do código de retorno **cruzado** com o marcador na saída.
Se os dois discordarem, o resultado é UNKNOWN em vez de um palpite — divergência
entre eles é sinal de que a suposição sobre a ferramenta está errada.
"""

import os
import shutil
import signal
import subprocess
import time

# ─── status ──────────────────────────────────────────────────────────────────

SAFE = "SAFE"                  # propriedade provada
UNSAFE = "UNSAFE"              # contraexemplo encontrado
PARSE_ERROR = "PARSE_ERROR"    # nao compilou / arquivo ausente
USAGE_ERROR = "USAGE_ERROR"    # flag invalida — o ESBMC nem rodou
TIMEOUT = "TIMEOUT"            # estourou o tempo; indeciso
UNKNOWN = "UNKNOWN"            # rc e saida discordam, ou rc inesperado

#: Status em que NENHUMA verificação ocorreu. Tratar qualquer um deles como
#: resultado de verificação é erro de interpretação.
NAO_VERIFICOU = (PARSE_ERROR, USAGE_ERROR, TIMEOUT, UNKNOWN)


# ─── localização do binário ──────────────────────────────────────────────────

def _find_esbmc():
    """$ESBMC_BIN -> $PATH -> build local -> binário embarcado no repo."""
    env = os.environ.get("ESBMC_BIN")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("esbmc")
    if found:
        return found
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in (("..", "..", "build", "src", "esbmc", "esbmc"),
                ("..", "QNNVerifier", "esbmc-6.8.0", "esbmc")):
        cand = os.path.abspath(os.path.join(here, *rel))
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


ESBMC_BIN = _find_esbmc()


class VerificationResult:
    def __init__(self, status, stdout, stderr, time_taken, returncode=None,
                 cmd=None):
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        self.time_taken = time_taken
        self.returncode = returncode
        self.cmd = cmd

    @property
    def output(self):
        """stdout + stderr. A 8.0.0 manda quase tudo para stderr por padrão,
        então parsear só stdout perde o trace."""
        return self.stdout + self.stderr

    @property
    def is_safe(self):
        return self.status == SAFE

    @property
    def is_unsafe(self):
        return self.status == UNSAFE

    @property
    def verificou(self):
        """True somente se o ESBMC chegou a emitir um veredito."""
        return self.status in (SAFE, UNSAFE)

    def raise_if_nao_verificou(self):
        if not self.verificou:
            raise RuntimeError(
                f"ESBMC nao verificou: status={self.status} rc={self.returncode}\n"
                f"cmd: {' '.join(self.cmd or [])}\n"
                f"{self.output[-2000:]}"
            )
        return self

    def __repr__(self):
        return (f"<VerificationResult {self.status} rc={self.returncode} "
                f"{self.time_taken:.2f}s>")


# ─── construção da linha de comando ──────────────────────────────────────────

def build_esbmc_cmd(filepath, esbmc_bin=None, **kwargs):
    """Monta os argumentos do ESBMC.

    Atenção às flags que NÃO existem: `--bounds-check` e `--pointer-check`
    foram removidas porque o ESBMC não as reconhece (`unrecognised option`,
    rc=64). Esses checks são ligados por PADRÃO; a ferramenta só expõe as formas
    negativas, mapeadas aqui como `no_bounds_check` / `no_pointer_check`.
    """
    cmd = [esbmc_bin or ESBMC_BIN, filepath]

    for key, flag in (
        # solvers
        ("z3", "--z3"), ("bitwuzla", "--bitwuzla"), ("mathsat", "--mathsat"),
        ("boolector", "--boolector"), ("cvc", "--cvc"),
        # codificação
        ("floatbv", "--floatbv"), ("fixedbv", "--fixedbv"),
        # laços e asserções
        ("no_unwinding_assertions", "--no-unwinding-assertions"),
        ("k_induction", "--k-induction"),
        ("incremental_bmc", "--incremental-bmc"),
        # checagens (formas NEGATIVAS: as positivas sao o padrao)
        ("memory_leak_check", "--memory-leak-check"),
        ("overflow_check", "--overflow-check"),
        ("no_bounds_check", "--no-bounds-check"),
        ("no_pointer_check", "--no-pointer-check"),
        # pipeline
        ("multi_property", "--multi-property"),
        ("smt_formula_only", "--smt-formula-only"),
    ):
        if kwargs.get(key):
            cmd.append(flag)

    if kwargs.get("unwind") is not None:
        cmd.extend(["--unwind", str(kwargs["unwind"])])
    if kwargs.get("max_k_step") is not None:
        cmd.extend(["--max-k-step", str(kwargs["max_k_step"])])
    cmd.extend(kwargs.get("extra_args") or [])
    return cmd


# ─── execução ────────────────────────────────────────────────────────────────

def _classify(returncode, out):
    """Cruza o código de retorno com o marcador da saída.

    Discordância vira UNKNOWN de propósito: significa que a premissa sobre a
    ferramenta está errada, e adivinhar aí é como o defeito original nasceu.
    """
    ok = "VERIFICATION SUCCESSFUL" in out
    failed = "VERIFICATION FAILED" in out
    if returncode == 0 and ok:
        return SAFE
    if returncode == 1 and failed:
        return UNSAFE
    if returncode in (6, 7):
        return PARSE_ERROR
    if returncode == 64:
        return USAGE_ERROR
    return UNKNOWN


def run_esbmc(filepath, timeout=60, **kwargs):
    """Roda o ESBMC e devolve um VerificationResult com status tri-estado."""
    if ESBMC_BIN is None:
        raise FileNotFoundError(
            "Binário do ESBMC não encontrado. Defina $ESBMC_BIN, coloque `esbmc` "
            "no PATH, compile em build/src/esbmc/, ou use o binário embarcado em "
            "pibic/QNNVerifier/esbmc-6.8.0/esbmc."
        )

    cmd = build_esbmc_cmd(filepath, **kwargs)
    start = time.time()

    # start_new_session: o ESBMC faz fork em --k-induction-parallel; sem grupo
    # de processos o timeout mata só o pai e deixa os filhos rodando solvers.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",  # traces nao sao UTF-8 puro
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        elapsed = time.time() - start
        return VerificationResult(_classify(proc.returncode, out + err),
                                  out, err, elapsed, proc.returncode, cmd)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
        return VerificationResult(TIMEOUT, out or "", err or "",
                                  time.time() - start, None, cmd)
