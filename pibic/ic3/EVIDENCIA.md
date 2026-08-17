# Evidência bruta das medições da raia E

Saídas de solver preservadas verbatim. Existem porque `KB-D03` registra que
**nenhum log em `results/` contém um veredito** — o projeto reportou provas cujos
artefatos não foram commitados. Um board que faz essa acusação precisa commitar
os seus.

## Ambiente

| | |
|---|---|
| Modelo | Ator DDPG real, 4→24→24→1, 745 parâmetros Q8.8 |
| Fonte dos pesos | `cartpole/webapp/public/ddpg_weights_q88.json` |
| Fidelidade | `validate_forward.py` — 12/12 estados bit-exatos vs. referência Python |
| Geração | `gen_transition_system.py --bits 16` |
| Síntese | yosys 0.33 → **904.242 portas AND, 65 latches, nível 689** |
| Motor | `yosys-abc` (ABC embutido no yosys 0.33) |
| Propriedade | `\|th\| ≤ 53` em Q8.8 = 12°, a partir de `\|estado\| ≤ 5` |
| Máquina | Linux x86_64, 15 GB RAM |

## Resultados

| Arquivo | Comando | Veredito | Tempo | Pico RSS |
|---|---|---|---|---|
| `cl_ddpg16.bmc.out` | `bmc3 -F 60 -T 900` | 5 frames, sem veredito | 901,4 s | 503 MB |
| — | `pdr -v -T 1800` | não convergiu | 1802,4 s | 708 MB |

**`Timeout` é indeciso, não "seguro".** Nenhuma das duas execuções decide a
propriedade.

## Lacuna conhecida

A saída bruta do PDR (`cl_ddpg16.abc.out`) **não existe**: aquela execução ocorreu
antes de `run_pdr.sh` passar a preservar o stdout do ABC. O filtro por palavras-chave
não casou nada e descartou o resto, então o detalhe do que o ABC fazia ao parar se
perdeu. Restam apenas tempo e pico de memória, medidos pelo wrapper Python.

Corrigido em `ad995812` — a saída bruta agora é sempre salva. Reproduzir com:

    ./run_pdr.sh cl_ddpg16.v 1800

Registrar a lacuna em vez de apagá-la é o mesmo critério que o board aplica ao
projeto: ausência de evidência se declara, não se silencia.
