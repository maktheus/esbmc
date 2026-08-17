# Kanban — Verificação Formal de IA com ESBMC (PIBIC)

Backlog derivado da auditoria de código de 2026-08-17. Cada tarefa carrega a
evidência que a originou (`arquivo:linha`) e o nível de confiança dessa evidência.

> **Este arquivo é a fonte de verdade do backlog.** O `prd.json` na raiz de `pibic/`
> está corrompido — contém 200 tarefas de *"CLI (Cover Display) Framework"* (Android
> foldable), não deste projeto. Ver `KB-F05`.

---

## O que está sendo feito

**O problema que esta auditoria encontrou não é código ruim — é evidência quebrada.**
O projeto tem trabalho sólido dentro dele (a camada de quantização Q8.8 é bit-a-bit
consistente entre Python, TypeScript e C, e o `quantization_report.json` reproduz até
o 10º dígito). Mas o artigo afirma resultados que os arquivos do repositório não
sustentam, e em alguns casos contradizem.

Por isso a disciplina central deste board: **cada tarefa carrega a procedência da sua
evidência.** Uma afirmação verificada por execução (✅) e uma relatada por agente sem
confirmação (🟡) não são a mesma coisa, e tratá-las igual foi exatamente como o
problema se instalou. Os agentes de auditoria, por exemplo, afirmaram que não existia
`.github/` no repositório — existe, com 13 workflows, um deles o `esbmc-verify.yml`
que nunca rodou. Se aquilo tivesse virado tarefa sem checagem, o esforço iria para o
lugar errado.

### Como o board foi produzido

1. **Cinco agentes de auditoria read-only em paralelo**, um por área, com posse
   exclusiva de arquivos. Três retornaram; dois se perderam sem notificação — as
   áreas deles estão registradas na raia A como ⬜ não auditado, não como "ok".
2. **Verificação direta** de toda afirmação de alto impacto, com comando e saída
   registrados. É o que separa os 31 ✅ dos 19 🟡.
3. **Medição, não estimativa**, onde havia número em jogo: a parede do BMC e o
   resultado do IC3 na raia E vêm de execução cronometrada, não de extrapolação.

### Estado agora

| | |
|---|---|
| **Concluído** | `KB-C00` — as duas vacuidades P0 (Properties B e C) confirmadas por reprodução direta, sem rede neural. `KB-F08` — submódulo `famous_pid` registrado, biblioteca recuperada. `KB-E01` — pipeline IC3 versionado em `pibic/ic3/`. |
| **Em execução** | `KB-A01` e `KB-A02` (as duas auditorias perdidas, relançadas). `KB-E02` — PDR sobre o ator DDPG real, forward já validado bit-a-bit (12/12 estados). |
| **Aberto, decisão do autor** | `KB-D02` — a Arduino-PID foi recuperada, mas o harness não compila por três motivos independentes; fechar exige compilar a ESBMC 8.0.0 ou retirar a afirmação. |

### O que a raia E significa

A verificação de malha fechada do cartpole é de **1 passo**, e o harness de 50 passos
nunca foi implementado. A leitura fácil seria falta de tempo. A medição mostra outra
coisa: 4 passos custaram 909 s e 383 MB, e o custo explode. **É limite do método, não
do esforço.** Isso transforma a maior limitação do trabalho em contribuição — desde que
a comparação seja feita com honestidade.

Duas ressalvas que valem mais que o resultado: os 0,37 s de prova ilimitada foram sobre
um **modelo sintético** com aritmética diferente da do projeto (soma-depois-divide, uma
camada), e no modelo *sem batente físico* o PDR **também não convergiu**. A vantagem do
IC3 é estrutural — memória independente da profundidade da prova, e um invariante
indutivo em vez de "seguro até K" — não é vantagem incondicional. O número que vale para
o artigo é o de `KB-E02`, sobre o ator real, e ainda está sendo medido.

---

## Legenda

**Status:** `todo` · `doing` · `blocked` · `done`

**Prioridade:**

| | Significado |
|---|---|
| **P0** | Invalida uma conclusão publicada. Resolver antes de qualquer submissão. |
| **P1** | Quebra reprodutibilidade ou integridade de dados. |
| **P2** | Higiene, dívida técnica, escopo prometido e não entregue. |

**Confiança na evidência:**

| | Significado |
|---|---|
| ✅ | **Verificado por execução direta** nesta auditoria. Comando e saída registrados. |
| 🟡 | **Relatado por agente**, não confirmado independentemente. Confirmar antes de agir. |
| ⬜ | **Não auditado.** Nenhum agente cobriu esta área (ver raia A). |

---

## Raia A — Completar a cobertura da auditoria

Dois dos cinco agentes de auditoria não retornaram. Estas áreas seguem sem revisão.

| ID | Prio | Tarefa | Evidência | Conf. | Status |
|---|---|---|---|---|---|
| KB-A01 | P1 | Auditar `cases/` — 4 verticais + `llm_ffn_verification/` (geradores FFN GPT-2/Llama, LUTs GELU/SiLU, reprodutibilidade de `verify_output/`) | agente perdido | ⬜ | `todo` |
| KB-A02 | P1 | Auditar harnesses de NN — `verification/`, `models/`, `deepseek_moe_verification/`, `caso_inicial_de_avaliacao/` | agente perdido | ⬜ | `todo` |
| KB-A03 | P2 | Decidir o papel de `QNNVerifier/` — dependência de terceiros vendorizada ou código morto? | 308 MB de 322 MB do diretório | ✅ | `todo` |

---

## Raia B — Núcleo Python & CI

| ID | Prio | Tarefa | Evidência | Conf. | Status |
|---|---|---|---|---|---|
| KB-B01 | P0 | **`run_esbmc` não lê `returncode`** — todo modo de erro (flag inválida, parse error, arquivo ausente) colapsa em `is_safe=False`, que é o valor que significa "bug encontrado". Substituir por status tri-estado `{SAFE, UNSAFE, PARSE_ERROR, USAGE_ERROR, TIMEOUT, UNKNOWN}` | `core_verify/esbmc_caller.py:62-67` | 🟡 | `todo` |
| KB-B02 | P0 | **`--bounds-check` e `--pointer-check` não existem no ESBMC.** Confirmado: `ESBMC error: unrecognised option '--bounds-check'` (exit 64). Esses checks são ligados por padrão; só existem as formas negativas. Enquanto não corrigido, os testes que os usam nunca executam verificação | `tests/test_chaos.py:18`, `tests/test_inference.py:25` | ✅ | `todo` |
| KB-B03 | P1 | Caminho do binário rígido para `../../build/src/esbmc/esbmc`, que não existe. Sem fallback para `$PATH`, `$ESBMC_BIN`, ou o binário 6.8.0 presente no próprio repo | `core_verify/esbmc_caller.py:6` | ✅ | `todo` |
| KB-B04 | P1 | **CI nunca rodou.** `esbmc-verify.yml` dispara em `main`; o branch default é `master`. Além disso, `export PATH` em um step não persiste para o próximo no GitHub Actions, e o step se chama literalmente *"(Mocked from latest release)"* | `.github/workflows/esbmc-verify.yml:3-7,25` | ✅ | `todo` |
| KB-B05 | P1 | Adicionar `ast2json` a `requirements.txt` — sua ausência é a causa única da falha do Caso 1 | `results/case1_mlp.log` | ✅ | `todo` |
| KB-B06 | P1 | Parser de contraexemplos: regex casam em linhas `PASSED:` (falso positivo em verificação bem-sucedida); `"NaN or Inf"` não existe na saída do ESBMC (é `"NaN on "`); `memory leak` engloba `dereference failure` | `core_verify/SMT_feedback_parser.py:14-17,38-40` | 🟡 | `todo` |
| KB-B07 | P1 | `text=True` sem `errors=` derruba o processo com `UnicodeDecodeError` em traces não-UTF-8 | `core_verify/esbmc_caller.py:62` | 🟡 | `todo` |
| KB-B08 | P1 | Adicionar casos ground-truth **pareados** (safe/unsafe). Hoje só há asserções negativas (`assert not is_safe`), satisfeitas por qualquer erro — por isso `KB-B01` é indetectável pela suíte | `roadmap.md:64-67` | 🟡 | `todo` |
| KB-B09 | P2 | `eval "$VERIFY_CMD"` executa string vinda de JSON que o próprio agente LLM edita | `ralph_loop_esbmc.sh:62` | ✅ | `todo` |
| KB-B10 | P2 | `pyproject.toml` não instala: flat-layout com múltiplos pacotes top-level, e `readme = "README.md"` aponta para arquivo inexistente | `pyproject.toml:9` | 🟡 | `todo` |
| KB-B11 | P2 | Timeout não mata o grupo de processos (sem `start_new_session` / `killpg`), apesar de `roadmap.md:5.2` prometer isso | `core_verify/esbmc_caller.py:69-73` | 🟡 | `todo` |

---

## Raia C — Propriedades formais

**`KB-C01` e `KB-C02` estão confirmadas** — reproduzidas por execução direta, não são
mais relato de agente. Os demais itens 🟡 desta raia seguem sem confirmação
independente: nenhuma correção deve ser feita sem reproduzir o diagnóstico primeiro.

| ID | Prio | Tarefa | Evidência | Conf. | Status |
|---|---|---|---|---|---|
| KB-C00 | P0 | **Reproduzir os diagnósticos de vacuidade** de `KB-C01` e `KB-C02`. Método: rodar o assert isolado, sem a rede neural. **Feito** — ambos confirmados, ver as duas linhas abaixo | `scratchpad/kbc00/propB_sem_rede.c`, `propC_sem_rede.c` | ✅ | `done` |
| KB-C01 | P0 | **Property B é vácua — confirmado.** O assert é sobre `th_new = th + (5*thd)/256`, que depende só de `th` e `thd`, ambas entradas livres; `F_Q` não aparece. `F_Q` alimenta apenas `th_acc → thd_new`, nunca asseridos. Rodei sem rede alguma e com `F_Q` arbitrário: `VERIFICATION FAILED` em **0,16 s**, contraexemplo `th=53, thd=64 → th_new=54`. Os 90 s e o contraexemplo publicados não dizem nada sobre o controlador. Reescrever o assert para depender de `F_Q` (ex.: sobre `thd_new`, ou sobre `th` após 2 passos) e, até lá, remover a Property B do webapp | `cartpole/verify_ddpg_closed_loop.py:185-190` | ✅ | `todo` |
| KB-C02 | P0 | **Property C é vácua — confirmado.** `tanh_abs` satura em 255 nos cinco ramos da aproximação, logo `F_Q = (255*10*256)/256 = 2550 < 2560` **por construção**, independente dos pesos. Rodei com `z = nondet_int()` livre, sem rede: `VERIFICATION SUCCESSFUL` em **0,72 s**. Reclassificar como sanity check da quantização — não é prova sobre o controlador. Bônus: `(tanh_z * 10 * 256) / 256` multiplica e divide por 256 sem efeito | `cartpole/verify_ddpg_closed_loop.py:226`; `TANH_APPROX_C` em `:70-80` | ✅ | `todo` |
| KB-C03 | P1 | Property A: trocar `assert(z > 0)` por `assert(F_Q > 0)`. A equivalência `z>0 ⟺ F>0` falha em Q8.8 porque `tanh_q88(1) = 0` | `cartpole/verify_ddpg_closed_loop.py:113,143` | 🟡 | `todo` |
| KB-C04 | P1 | `pre_bound` fixos (2048/4096) seriam arbitrários, não derivados. Substituir por `interval_propagate_layer`, já usado no closed-loop. Adicionar assert de sanidade que falhe se o `assume` for insatisfazível — hoje vacuidade seria interpretada como "neurônio morto" | `cartpole/verify_ddpg_dead_neurons.py:227,233` | 🟡 | `todo` |
| KB-C05 | P1 | Veredito "vivo" da camada 2 não seria sound (`h1` como caixa relaxada). Enquanto não corrigido, reportar "24/48 provados", não "0/48 mortos" | `cartpole/verify_ddpg_dead_neurons.py:105-139` | 🟡 | `todo` |
| KB-C06 | P1 | Habilitar `--overflow-check` em ao menos uma execução por harness. Nenhuma das 6 invocações usa — ausência de overflow é assumida, não provada | 6 chamadas em `cartpole/verify_*.py` | 🟡 | `todo` |
| KB-C07 | P2 | Unificar a física: `webapp/lib/physics.ts` teria divergido de `cartpole_env.py` (parede com restituição, critério de falha só em θ). Treino, verificação e demonstração usariam três plantas diferentes | `webapp/lib/physics.ts:35-56` vs `cartpole_env.py:59-68` | 🟡 | `todo` |

---

## Raia D — Evidências & artigo

| ID | Prio | Tarefa | Evidência | Conf. | Status |
|---|---|---|---|---|---|
| KB-D01 | P0 | **Tabela GEMM contradiz o único dado medido.** Artigo: N=2 → `<1s`, N=3 → `≈2s`. CSV real: `4.5900` e `53.2923` (9× e 27×). N=4, 5, 6 não existem em lugar nenhum. Reexecutar e logar, ou remover a tabela | `artigo/caps/4resultados.tex:176-190` vs `results/case2_benchmark.csv` | ✅ | `todo` |
| KB-D02 | P0 | **Verificação da Arduino-PID-Library afirmada sem qualquer artefato.** `verification/famous_pid/` aparece vazio porque é um **submódulo quebrado**: está no índice como gitlink (modo `160000`, commit `524a4268`) mas **não existe `.gitmodules`** no repositório. **`KB-F08` recuperou a biblioteca, e a tentativa de reexecução revelou que o harness nunca pôde ter rodado.** Tentei `verify_pid.cpp` com as flags exatas do artigo (`--floatbv --unwind 11`) e encontrei **três bloqueios independentes**, cada um suficiente para impedir a execução: (1) o submódulo quebrado tornava `#include "famous_pid/PID_v1.h"` irresolvível — corrigido por `KB-F08`; (2) a macro `ARDUINO` **não é definida em lugar nenhum do repo**, então `PID_v1.cpp:8` cai no ramo pré-Arduino-1.0 e pede `WProgram.h`, que não existe (o mock `Arduino.h` só declara `millis()`); (3) `PID_v1.cpp:46` usa **construtor delegante C++11**, e o binário 6.8.0 embarcado **não tem a flag `--std`** (só existe na 8.0.0, `src/esbmc/options.cpp:136`) — testei `--std c++11`, `-std=c++11` e `--std=c++11`, todos `unrecognised option`. Conclusão: a afirmação de `VERIFICATION SUCCESSFUL` **não é reproduzível com o ferramental do repositório**. Para fechar: compilar a 8.0.0 e rodar com `--std c++11`, ou remover a afirmação | `artigo/caps/4resultados.tex:252-256`; execução registrada | ✅ | `todo` |
| KB-D03 | P0 | **Nenhum log em `results/` contém `VERIFICATION SUCCESSFUL`.** Toda afirmação de prova bem-sucedida no capítulo 4 está sem evidência | `grep -rl` em `results/` | ✅ | `todo` |
| KB-D04 | P0 | Remover as 6 legendas *"Elaborada pelo autor (Subagente Grafico/Curvas/Arquitetural/de Controle/RL)"* e *"sugerida pelo Subagente Metodológico"* | `4resultados.tex:73,117,217,245,308`; `3metodologia.tex:130` | ✅ | `todo` |
| KB-D05 | P0 | **Caso 3 não produziu dados.** `case3_agent_stats.csv` e `case3_console.log` têm 0 bytes. Causa provável: `--smtlib` faz o ESBMC emitir a fórmula em vez de resolvê-la, então `"VERIFICATION SUCCESSFUL" in stdout` é sempre falso | `results/case3_*` (0 bytes); `3_neuro_symbolic/mock_agent.py:63,67` | ✅ | `todo` |
| KB-D06 | P0 | Corrigir a afirmação de CI ativa no artigo — ou implementar a CI (ver `KB-B04`) | `artigo/caps/3metodologia.tex:258-260` | ✅ | `todo` |
| KB-D07 | P1 | Instalar `ast2json` e reexecutar o Caso 1. O log atual termina em `ERROR: Module 'ast2json' not found` — a verificação de Python, pilar declarado do trabalho, nunca rodou | `results/case1_mlp.log` | ✅ | `todo` |
| KB-D08 | P1 | **Criar `pibic/README.md`.** Não existe. Precisa de: setup do ESBMC (escolher entre binário embarcado e build from source), ordem de execução end-to-end, mapa dos diretórios, versões, como compilar o artigo | — | ✅ | `todo` |
| KB-D09 | P1 | Registrar ambiente por experimento (versão ESBMC, solver, flags, timeout, CPU/RAM, seed). Hoje há 4 combinações conflitantes no repo: 8.0.0/Z3, 6.8.0/Boolector 3.2, 6.8.0/Z3 4.8.9, "compilada do fonte". Usar `cartpole/ESBMC_NOTES.md` como modelo — é o melhor documento do repositório | `4resultados.tex:4` vs `apresentacao_pibic.tex:593` | 🟡 | `todo` |
| KB-D10 | P1 | Religar as figuras aos dados. Nenhum script em `artigo/figs/` abre arquivo algum — todos os valores são literais. `plot_case2.py:7-8` tem a tabela GEMM inventada que contradiz o CSV | `artigo/figs/*.py` | 🟡 | `todo` |
| KB-D11 | P1 | Corrigir caminho absoluto `/home/uchoa/...` em 9 arquivos — quebra a execução em qualquer outra máquina, e 2 imagens do `analysis_report.md` não renderizam para ninguém | `plot_pipeline_esbmc.py:40`, `results/analysis_report.md:118,122`, +6 | ✅ | `todo` |
| KB-D12 | P2 | **Reconstruir `referencias.bib`.** 18 entradas, 2 citadas. 13 são de outro trabalho (futebol/visão computacional: `offside`, `soccer`, `yolo`). O artigo **não cita o ESBMC**, sua ferramenta central | `artigo/referencias.bib` | ✅ | `todo` |
| KB-D13 | P2 | Incorporar ao artigo o trabalho feito e não reportado: cart-pole DQN/DDPG (34 arquivos, contraexemplos concretos) e FFN GPT-2/Llama (speedup 18–27×). É o material mais rigoroso do repositório e está ausente do capítulo 4 | — | 🟡 | `todo` |
| KB-D14 | P2 | Substituir stubs do template abnTeX2: `an_1.tex`, `ap_1.tex`, `errata.tex` (errata sobre neoplasias em cães), `siglas.tex`, `simbolos.tex`, e placeholders "Nome Completo da Pessoa"/"CCHE/UEPB" | `artigo/anexos_apendices/`, `artigo/editar/` | 🟡 | `todo` |
| KB-D15 | P2 | Fechar `test.md`: os 3 itens (`poisoning`, `propriedade nova para regressão`, `DS verifier`) estão integralmente não iniciados. Ou entram no plano com escopo, ou saem | `test.md` | 🟡 | `todo` |

---

## Raia E — IC3/PDR (nova capacidade)

Origem: a verificação de malha fechada é de **1 passo**, e o harness de 50 passos
(`cartpole/closedloop_esbmc_stub.c`) nunca foi implementado. A medição mostra que
isso é um limite do método, não do esforço.

> ⚠️ **A tabela abaixo é de um modelo SINTÉTICO, não do DDPG do projeto.** São 24
> neurônios em uma camada, pesos aleatórios de seed 7, e — a diferença que importa —
> aritmética **soma-depois-divide** (`(w₀·x + w₁·ẋ + …)/256`). O ator real tem duas
> camadas de 24 (745 parâmetros) e faz **divisão por termo**
> (`(x·w₀)/256 + (ẋ·w₁)/256 + …`), com truncamento em cada produto. São
> controladores diferentes. Os números servem para caracterizar a *parede do BMC*,
> que é o ponto, mas **não são resultado sobre o controlador do projeto** — esse é o
> objeto de `KB-E02`.

**Dados medidos nesta auditoria** (modelo sintético: 24 neurônios ReLU, seed 7, Q8.8):

| Método | Modelo | Veredito | Tempo | Pico RSS |
|---|---|---|---|---|
| ESBMC BMC | 32-bit, K=1 | SUCCESSFUL *(1 passo)* | 0,3 s | 46 MB |
| ESBMC BMC | 32-bit, K=2 | SUCCESSFUL *(2 passos)* | 77 s | 137 MB |
| ESBMC BMC | 32-bit, K=4 | SUCCESSFUL *(4 passos)* | 909 s | 383 MB |
| ABC BMC | 16-bit | 5 frames, sem veredito | 240 s | — |
| ABC PDR | 16-bit | não convergiu | 500 s | 110 MB |
| **ABC PDR** | **16-bit + batente** | **PROVADO ∀t** | **0,37 s** | trivial |

Invariante encontrado: **4 cláusulas, 8 literais, 3 dos 65 bits de estado** —
`th[13] ⟺ th[14] ⟺ th[15]`. Convergiu em `F[2]`.

| ID | Prio | Tarefa | Evidência | Conf. | Status |
|---|---|---|---|---|---|
| KB-E01 | P1 | **Feito** — pipeline versionado em `pibic/ic3/` (commit `25d02b9f`): `gen_transition_system.py` (lê os pesos reais e emite o sistema de transição Verilog, `--bits 16|32`), `validate_forward.py` (prova bit-exatidão do forward contra referência Python) e `run_pdr.sh` (yosys → AIGER → `abc pdr`, com tempo e pico de RSS). O Verilog é o ponto de bifurcação: dele saem tanto AIGER quanto BTOR2, então trocar de motor não exige reescrever o gerador | `pibic/ic3/` | ✅ | `done` |
| KB-E02 | P1 | **Gerador já lê os pesos reais** (`ddpg_weights_q88.json`, ator 4→24→24→1, 745 parâmetros) com a semântica exata do harness verificado: divisão **por termo**, truncamento para zero, ReLU, tanh de 5 ramos, dinâmica linearizada 4040/375. `validate_forward.py` confirma **12/12 estados exatos**, incluindo os dois cantos de fronteira. Diferença deliberada: os `__ESBMC_assume` de intervalo do harness C são **omitidos** — são muleta do BMC, e sem eles o modelo não descarta nenhum estado alcançável (ver `KB-C04`). Propriedade: `\|th\| ≤ 53` (12°) para sempre. **Execução do PDR em andamento** | `pibic/ic3/validate_forward.py` | ✅ | `doing` |
| KB-E03 | P1 | Fechar a curva de escalabilidade do BMC com dado real: medir K=8 e K=16. **Não extrapolar** de 3 pontos — é exatamente o erro metodológico de `KB-D01` | `4resultados.tex:198` | ✅ | `todo` |
| KB-E04 | P2 | Documentar as limitações honestas: (a) o PDR também não convergiu no modelo sintético sem batente (500 s); (b) a vantagem é **estrutural** — memória independente da profundidade — não incondicional; (c) bit-blastar para AIGER **perde a estrutura de palavra**, o que atrapalha a generalização de cláusulas do PDR. Ver `KB-E07` | ✅ medido | ✅ | `todo` |
| KB-E05 | P2 | Avaliar `--overflow-check` / bit-width: 32→16 bits cortou o estado de 129 para 65 flops e a memória do PDR de 246 para 110 MB. O cartpole real é Q8.8 = 16 bits | ✅ medido | ✅ | `todo` |
| KB-E06 | P2 | Reescrever o Caso 5 do artigo (52 topologias, sem artefato) usando os dados de `KB-E03` — resultado medido no lugar de número inventado | `4resultados.tex:265` | ✅ | `todo` |
| KB-E07 | P2 | **Rota word-level como plano B.** Se o ABC não convergir no modelo real, exportar BTOR2 (`yosys write_btor`, já disponível) e usar um motor word-level — AVR, Pono ou `btormc`. Nenhum está instalado nem no apt (`btormc` vem com o Boolector; AVR e Pono exigem build do fonte). BTOR2 preserva os 4 inteiros em vez de 65 bits soltos, o que tende a ajudar muito a generalização | `yosys -p "help write_btor"`; `command -v avr pono btormc` → ausentes | ✅ | `todo` |

---

## Raia F — Higiene do repositório

| ID | Prio | Tarefa | Evidência | Conf. | Status |
|---|---|---|---|---|---|
| KB-F01 | P1 | **Mover 6 arquivos do PIBIC da raiz do fork do ESBMC** para `pibic/`: `chatbot_hibrido.py`, `run_chatbot.sh`, `generated_code.c`, `nn_desconto_esbmc.c`, `case2_output.txt` (875 KB), `case3_output.txt` (0 bytes). Contaminam a árvore upstream e complicam rebase | `git ls-files` na raiz | ✅ | `todo` |
| KB-F02 | P1 | Decidir sobre os binários em `QNNVerifier/esbmc-6.8.0/`: 71 MB (`esbmc`) + 57 MB (`esbmc.exe`) + 15 MB (`libz3.dll`). `pibic/` tem 322 MB, dos quais 308 MB são `QNNVerifier/` | `du -sh` | ✅ | `todo` |
| KB-F03 | P1 | Criar `pibic/.gitignore` — não existe. `tests/test_chaos.py` gera `.c` dentro da árvore de fontes | — | ✅ | `todo` |
| KB-F04 | P2 | Reconciliar as duas taxonomias concorrentes: `1_python_models/`…`4_control_system/` (legado) vs `cases/` + `core_verify/` (roadmap). Ambas coexistem com código duplicado. Declarar qual é canônica | `roadmap.md:37-56` | 🟡 | `todo` |
| KB-F05 | P2 | `prd.json` está corrompido — contém 200 tarefas de *"CLI (Cover Display) Framework"* (Android foldable). Regenerar de `generate_100_prd.py` ou aposentar em favor deste kanban. **Remover `recreate_prd.py`**, que causou o dano e aponta para `/home/uchoa/` | `prd.json`, `recreate_prd.py:4` | ✅ | `todo` |
| KB-F06 | P2 | Remover `teste_mlp/verify_mlp.c.draft` — conteria cadeia de raciocínio de LLM vazada (`// Wait, I noticed a typo in my thought`) e código que não compila | `teste_mlp/verify_mlp.c.draft` | 🟡 | `todo` |
| KB-F07 | P2 | Consolidar as variantes de `verify_mlp*.c` — declarar `verify_mlp_qnn.c` (gerado) como canônico e remover as demais | `teste_mlp/` | 🟡 | `todo` |
| KB-F08 | P1 | **Submódulo quebrado fazia o checkout do CI sair com código 128** — `pibic/verification/famous_pid` era gitlink `160000` sem `.gitmodules`. **Resolvido criando `.gitmodules`** apontando para `br3ttb/Arduino-PID-Library`: o commit `524a4268` existe upstream (br3ttb, 2024-05-31), então o conteúdo foi **recuperado, não descartado**. Verificado: `git submodule foreach` agora sai 0 (era 128); `git submodule update --init` faz checkout do commit exato e traz `PID_v1.cpp`/`PID_v1.h`; nenhum drift do gitlink | log do job 95281480323; `git submodule status` | ✅ | `done` |

---

# Organização em threads de agentes

## Princípio: raia = dono exclusivo de arquivos

O risco real ao paralelizar agentes **não é** o custo — é **dois agentes editando o
mesmo arquivo**. A tabela abaixo dá posse exclusiva de caminhos a cada raia. Um
agente só escreve dentro do seu conjunto.

| Raia | Escreve em | Não toca |
|---|---|---|
| **B** — Núcleo | `core_verify/`, `tests/`, `pyproject.toml`, `requirements.txt`, `.github/workflows/esbmc-verify.yml` | tudo mais |
| **C** — Propriedades | `cartpole/verify_*.py`, `cartpole/*.json`, `teste_mlp/`, `verification/` | `artigo/`, `core_verify/` |
| **D** — Evidências | `artigo/`, `apresentacao/`, `results/`, `*.md` na raiz de `pibic/` | código executável |
| **E** — IC3 | `pibic/ic3/` (novo, isolado) | tudo mais |
| **F** — Higiene | **deleções e movimentações em todo o repo** | — |

**Regra dura: a raia F nunca roda em paralelo com nada.** Deleções e `git mv`
conflitam com qualquer agente que esteja editando. F roda sozinha, entre ondas.

## Ondas de execução

```
ONDA 0  (sequencial, bloqueia tudo)
└── A01, A02 ....... completar a auditoria das áreas não cobertas
                     Sem isso, o backlog está incompleto e priorizar é chute.

ONDA 1  (3 agentes em paralelo — sem sobreposição de arquivos)
├── Agente B ....... B01→B03, B05, B07 ..... consertar o wrapper
├── Agente D ....... D01→D06 ............... inventário de afirmações sem evidência
└── Agente E ....... E01, E03 .............. versionar o pipeline IC3

ONDA 2  (sequencial — depende da Onda 1)
└── C00 ............ reproduzir os diagnósticos de vacuidade
                     Requer B01 pronto: sem status tri-estado, não dá para
                     distinguir "propriedade vácua" de "ESBMC nem rodou".

ONDA 3  (2 agentes em paralelo)
├── Agente C ....... C01→C06 ............... corrigir as propriedades
└── Agente B ....... B04, B06, B08 ......... CI + parser + ground-truth

ONDA 4  (sequencial)
└── Agente F ....... F01→F07 ............... higiene, sozinho

ONDA 5  (depende de C e E)
├── Agente E ....... E02, E04→E06 .......... IC3 com pesos reais
└── Agente D ....... D07→D15 ............... reescrever com dado medido
```

## Por que essa ordem

- **B antes de C.** Enquanto `run_esbmc` colapsar todo erro em `is_safe=False`
  (`KB-B01`), qualquer diagnóstico de propriedade é ambíguo: não dá para saber se
  a propriedade é vácua ou se o ESBMC abortou com flag inválida (`KB-B02`).
- **C antes de E02.** Provar uma propriedade por 50 passos com IC3 não vale nada se
  a propriedade não fala sobre o controlador. Corrigir vem antes de escalar.
- **D01–D06 podem começar já.** São remoções e reexecuções que não dependem de
  ninguém — e são as de maior risco reputacional.
- **F por último.** Deleções invalidam caminhos que outros agentes estão editando.

## Regras para quem despacha os agentes

1. **Um agente por raia, nunca dois na mesma.** Posse de arquivo é o mecanismo de
   exclusão mútua.
2. **Todo agente reporta evidência de execução**, não conclusão. "Rodei X, saiu Y"
   — não "está correto". A auditoria que gerou este backlog errou ao afirmar que
   `.github/` não existia; existe, com 13 workflows. Agentes se enganam.
3. **Nenhuma tarefa 🟡 vira commit sem virar ✅ antes.** Reproduzir o diagnóstico é
   parte da tarefa, não pré-requisito opcional.
4. **Agentes de auditoria são read-only.** Só as ondas de execução escrevem.
5. **Se um agente não retornar, a tarefa volta para `todo`.** Aconteceu com A01 e
   A02 nesta rodada — dois dos cinco agentes se perderam sem notificação.

## Snapshot

| Raia | P0 | P1 | P2 | Total |
|---|---|---|---|---|
| A — Auditoria pendente | — | 2 | 1 | 3 |
| B — Núcleo & CI | 2 | 6 | 3 | 11 |
| C — Propriedades | 3 | 4 | 1 | 8 |
| D — Evidências & artigo | 6 | 5 | 4 | 15 |
| E — IC3/PDR | — | 3 | 4 | 7 |
| F — Higiene | — | 4 | 4 | 8 |
| **Total** | **11** | **24** | **17** | **52** | **11** | **24** | **16** | **51** | **11** | **24** | **16** | **51** |

Confiança: **31 ✅ verificado** · **19 🟡 relatado por agente** · **2 ⬜ não auditado**

Contagens conferidas por script sobre as próprias linhas da tabela, não à mão — as
versões anteriores deste rodapé traziam 21/26/3 e 25/23/3, ambas erradas. Um board que
acusa números sem evidência não pode ter números sem evidência.

Progresso: `KB-C00` concluída — as duas vacuidades P0 (`KB-C01`, `KB-C02`) saíram de
🟡 para ✅ por reprodução direta. As auditorias `KB-A01` e `KB-A02` estão em execução.
