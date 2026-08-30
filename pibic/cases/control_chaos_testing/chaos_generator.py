"""
Injetor de ruído para o Caso 4 — perfis de caos como restrições SMT.

DOIS DEFEITOS CORRIGIDOS, ambos revelados quando o wrapper passou a distinguir
"erro de execução" de "propriedade violada":

  1. **O ruído injetado nunca chegava ao sistema.** Cada perfil declarava uma
     variável com nome próprio (`noise_uniform`, `noise_drift`, …), mas o
     template usava `float noise = 0.0f;` e somava *essa*. Os cinco perfis
     verificavam o mesmo sistema **sem ruído nenhum** — vacuidade da mesma
     família das Properties B e C do cartpole. Agora cada perfil declara
     `noise` diretamente, então a restrição alcança `measured_sensor`.

  2. **Literais C inválidos.** `f"{max_time}f"` com `max_time=100` (int)
     produzia `100f`, que não é literal válido — o perfil Drift falhava com
     PARSE_ERROR. `_f()` abaixo formata sempre com parte fracionária.
"""


def _f(v) -> str:
    """Literal float válido em C: 100 -> '100.0f', -0.5 -> '-0.5f'."""
    return f"{float(v):.6g}".rstrip("f") + ("f" if "." in f"{float(v):.6g}"
                                            else ".0f")


class ChaosGenerator:
    """Distribuições matemáticas mapeadas para restrições do ESBMC.

    Cada método devolve C que declara `noise` — o mesmo nome que o template
    soma em `measured_sensor`. Se um perfil declarar outro nome, a restrição
    fica órfã e o teste vira vácuo, que foi exatamente o defeito anterior.
    """

    @staticmethod
    def get_uniform_noise(bound=1.0):
        return (f"float noise = nondet_float();\n"
                f"    __ESBMC_assume(noise >= -{_f(bound)} && noise <= {_f(bound)});")

    @staticmethod
    def get_gaussian_noise(mean=0.0, std=1.0):
        # aproximacao limitada: Box-Muller exato estoura os limites nao-lineares
        b = std * 3.0  # intervalo de ~99%
        return (f"float noise = nondet_float();\n"
                f"    __ESBMC_assume(noise >= {_f(mean - b)} && noise <= {_f(mean + b)});")

    @staticmethod
    def get_impulse_noise(magnitude=100.0, probability=0.01):
        return (f"float noise = nondet_float();\n"
                f"    __ESBMC_assume(noise == 0.0f || noise == {_f(magnitude)}"
                f" || noise == {_f(-magnitude)});")

    @staticmethod
    def get_drift_noise(rate=0.1, max_time=100):
        return (f"float noise = nondet_float();\n"
                f"    __ESBMC_assume(noise >= 0.0f && noise <= {_f(rate * max_time)});")

    @staticmethod
    def get_sinusoidal_noise(amplitude=1.0):
        return (f"float noise = nondet_float();\n"
                f"    __ESBMC_assume(noise >= -{_f(amplitude)} && noise <= {_f(amplitude)});")


PERFIS = {
    "Uniform": ChaosGenerator.get_uniform_noise,
    "Gaussian": ChaosGenerator.get_gaussian_noise,
    "Impulse": ChaosGenerator.get_impulse_noise,
    "Drift": ChaosGenerator.get_drift_noise,
    "Sinusoidal": ChaosGenerator.get_sinusoidal_noise,
}


def inject_pid_chaos(template_path, output_path, noise_type="Uniform"):
    if noise_type not in PERFIS:
        raise KeyError(f"perfil desconhecido: {noise_type!r}; "
                       f"conhecidos: {sorted(PERFIS)}")

    with open(template_path) as fh:
        content = fh.read()

    marcador = "// {{NOISE_INJECTION}}"
    if marcador not in content:
        raise ValueError(f"{template_path} não contém {marcador}")

    content = content.replace(marcador, PERFIS[noise_type]())

    if "float noise " not in content:
        raise ValueError("injeção não declarou `noise` — o ruído ficaria órfão")

    with open(output_path, "w") as fh:
        fh.write(content)
    return output_path
