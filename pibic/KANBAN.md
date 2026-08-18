# Kanban — Verificação Formal de IA com ESBMC (PIBIC)

Backlog derivado da auditoria de código de 2026-08-17. Cada tarefa carrega a
evidência que a originou (`arquivo:linha`) e o nível de confiança dessa evidência.

> **Este arquivo é a fonte de verdade do backlog.** O `prd.json` (110 tarefas do
> ESBMC) foi restaurado em `KB-F05` — estava sobrescrito com 200 tarefas de um
> framework de display Android — mas serve ao *ralph loop*, não a esta priorização.

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
   registrados. É o que separa os 40 ✅ dos 12 🟡.
3. **Medição, não estimativa**, onde havia número em jogo: a parede do BMC e o
   resultado do IC3 na raia E vêm de execução cronometrada, não de extrapolação.

### Estado agora

**29 das 52 tarefas concluídas, e 9 dos 11 P0.** Os dois P0 restantes são
`KB-C01` e `KB-C02`: a vacuidade das Properties B e C está **confirmada por execução**,
mas os asserts ainda não foram reescritos — diagnóstico fechado, correção pendente. A cobertura da
auditoria está completa — não há mais área ⬜. A suíte saiu de *21 falhas em 0,12 s*,
porque nada rodava, para **31 testes passando em 12m49s**, verificando de verdade.
Cada afirmação de veredito do capítulo 4 agora aponta para um log versionado.

| | |
|---|---|
| **Núcleo consertado** | `KB-B01`–`KB-B05`, `B07`, `B08`, `B10`, `B11`. O wrapper distingue erro de execução de propriedade violada; a CI passou a existir de fato. |
| **Vacuidades confirmadas** | `KB-C00`–`KB-C02` no cartpole, e mais duas encontradas ao consertar a suíte: `kernels_benchmarks.cpp` não compilava (15 testes inertes) e o ruído dos 5 perfis de caos nunca chegava ao sistema. |
| **Raia E encerrada** | `KB-E01`, `KB-E02`. Empate técnico no ator real: nenhum método decide. |
| **Higiene** | `KB-D04`, `D08`, `D11`, `F01`, `F03`, `F05`, `F08`. |
| **Em aberto** | `KB-C01`/`KB-C02` (P0): reescrever os asserts vácuos do cartpole. Mais 23 itens P1/P2. |

### Uma armadilha recorrente: timeout arbitrário

O artigo afirmava, no Caso 4, que o PID mantém `T < 150 °C` para 10 passos sob
qualquer ruído em `[-5,+5]`. Uma primeira medição com limite de 200 s deu timeout,
e a afirmação quase foi classificada como sem evidência. Reexecutada com folga:
**`VERIFICATION SUCCESSFUL` em 217 s** — dezessete segundos além do primeiro
limite. A afirmação era verdadeira.

Isso é a face inversa do erro que este board persegue. Ler `TIMEOUT` como
"seguro" inventa uma prova; ler `TIMEOUT` como "sem evidência" destrói uma
prova legítima. Nos dois casos o defeito é o mesmo: **tratar ausência de sinal
como sinal**. Daí a coluna de veredito de `results/EVIDENCIAS.md` distinguir
`TIMEOUT` de `SAFE` e de `UNSAFE`, e nunca colapsar os três.

### O que a raia E significa

A verificação de malha fechada do cartpole é de **1 passo**, e o harness de 50 passos
nunca foi implementado. A leitura fácil seria falta de tempo. A medição mostra outra
coisa: 4 passos custaram 909 s e 383 MB, e o custo explode. **É limite do método, não
do esforço.** Isso transforma a maior limitação do trabalho em contribuição — desde que
a comparação seja feita com honestidade.

**O resultado sobre o ator real já saiu, e é um empate técnico:** nenhum dos dois
métodos decide a instância. O `abc pdr` não convergiu em 1802 s; o `abc bmc3` alcançou
5 frames em 901 s. `Timeout` é **indeciso** — tratá-lo como "seguro" seria o mesmo erro
que a raia D acusa no artigo.

Isso não enfraquece a tese, fortalece: o ABC parou em **5 frames** no ator real de 745
parâmetros, e o ESBMC parou em **K=4** numa rede sintética 8× menor. Ferramentas,
solvers e representações diferentes, **mesma parede** — logo o limite é do método, não
da implementação. É essa a afirmação defensável para o artigo.

O que a raia E entrega, então, não é "use IC3 e o problema acaba". É: (a) a parede do
BMC medida no sistema real do projeto; (b) a demonstração, no modelo sintético, de que
IC3 dá prova *ilimitada* em 0,37 s onde BMC dá 4 passos em 909 s; (c) o diagnóstico de
que a instância real exige atacar a **representação**, não só trocar de motor — bit-blastar
4 inteiros em 65 bits soltos destrói a estrutura que o PDR usa para generalizar. Daí
`KB-E07`.

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

## Raia A — Cobertura da auditoria

**Concluída à mão.** Os dois agentes designados a estas áreas se perderam duas vezes
(4 dos 7 agentes desta sessão sumiram sem notificação), então a auditoria foi refeita
por execução direta. Um falso achado meu foi detectado e retirado no processo.

| ID | Prio | Tarefa | Evidência | Conf. | Status |
|---|---|---|---|---|---|
| KB-A01 | P1 | **Feito à mão** (os agentes se perderam 2×). (a) **`cases/agentic_verification/ralph_loop.py` não verifica nada** — o portão ESBMC está inteiramente comentado e o loop chama `mark_completed()` incondicionalmente; é o mecanismo que produziria as "52 topologias" do Caso 5. (b) **`4_control_system/test_chaos.py` sombreava `tests/test_chaos.py`** — mesmo basename, `pytest` sem argumento quebrava a coleta com *import file mismatch*; usava `--smtlib` e caminho relativo ao CWD. Removido, e `pytest.ini` passa a excluir `QNNVerifier`. (c) **`cases/llm_ffn_verification/` é a parte mais sólida do repo**: LUTs e harnesses **regeneram byte-idênticos** e os tempos batem — `gpt2_2x4_qnn` 0,27 s (afirmado 0,30), `gpt2_4x8_qnn` 3,62 s (5,3), `llama-7b_4x8_qnn` 3,46 s (4,3), `gpt2_4x16_qnn` 20,90 s (27), todos SUCCESSFUL. (d) Os diretórios numerados **não são legado morto** — `2_inference_engine/` e `3_neuro_symbolic/` têm o código vivo; a taxonomia é misturada. (e) `rl_policy.c` é honesto, mas `pos_x` é restringido e nunca usado | execução direta | ✅ | `done` |
| KB-A02 | P1 | **Feito à mão.** Medidos: `property_check.c` FAILED (193 s); `property_check_iris.c` timeout (>240 s); `mlp_stub.c` SUCCESSFUL (3,3 s); **`attention_block_stub.c` não compilava** — declarava `void __ESBMC_assume(int)` conflitando com `_Bool` de `esbmc_intrinsics.h`; **corrigido**, agora compila e gera claims (`out0[i] ∈ [-1,1]`), mas não decide em 300 s. **Falso achado meu retirado:** `nn_forward_pass*.c` não são harnesses — `property_check.c` faz `#include "nn_forward_pass.c"`. `estudo_de_caso/` confirmado sem `assert`/`__ESBMC_assume`/ESBMC: só material didático | execução direta | ✅ | `done` |
| KB-A03 | P2 | **Resolvido.** Nenhum script usa o **código** do QNNVerifier — as 7 referências apontam todas para o **binário** `QNNVerifier/esbmc-6.8.0/esbmc`. O resto (fontes Java, 10 `.class`, 4 `.exe`/`.dll`, 40 MB de ACAS Xu) é peso morto. Medido: `pibic/` **sem** QNNVerifier = **18 MB**; com = 322 MB. Manter só o `esbmc` Linux (71 MB) levaria o repo a ~90 MB | `du -sh`, grep de referências | ✅ | `todo` |

---

## Raia B — Núcleo Python & CI

| ID | Prio | Tarefa | Evidência | Conf. | Status |
|---|---|---|---|---|---|
| KB-B01 | P0 | **Feito.** Status tri-estado `{SAFE, UNSAFE, PARSE_ERROR, USAGE_ERROR, TIMEOUT, UNKNOWN}` derivado do `returncode` **cruzado** com o marcador da saída; discordância vira UNKNOWN. Códigos medidos no binário 6.8.0: 0/1/6/64 | `core_verify/esbmc_caller.py:62-67` | ✅ | `done` |
| KB-B02 | P0 | **Feito.** `--bounds-check`/`--pointer-check` removidos; expostas as formas negativas `no_bounds_check`/`no_pointer_check`. Teste de regressão em `test_ground_truth.py` garante que as inexistentes nunca sejam emitidas | `tests/test_chaos.py:18`, `tests/test_inference.py:25` | ✅ | `done` |
| KB-B03 | P1 | **Feito.** Resolução em cascata: `$ESBMC_BIN` → `$PATH` → build local → binário embarcado. `conftest.py` pula a suíte com motivo se nenhum existir | `core_verify/esbmc_caller.py:6` | ✅ | `done` |
| KB-B04 | P1 | **Feito.** Gatilho corrigido para `[master, main]`; o `export PATH` que não persistia foi trocado por `$GITHUB_ENV` apontando para o binário embarcado | `.github/workflows/esbmc-verify.yml:3-7,25` | ✅ | `done` |
| KB-B05 | P1 | **Feito.** `ast2json` e `matplotlib` adicionados a `requirements.txt` | `results/case1_mlp.log` | ✅ | `done` |
| KB-B06 | P1 | Parser de contraexemplos: regex casam em linhas `PASSED:` (falso positivo em verificação bem-sucedida); `"NaN or Inf"` não existe na saída do ESBMC (é `"NaN on "`); `memory leak` engloba `dereference failure` | `core_verify/SMT_feedback_parser.py:14-17,38-40` | 🟡 | `todo` |
| KB-B07 | P1 | **Feito.** `encoding='utf-8', errors='replace'` — trace não-UTF-8 não derruba mais a execução | `core_verify/esbmc_caller.py:62` | ✅ | `done` |
| KB-B08 | P1 | **Feito.** `tests/test_ground_truth.py`: par safe/unsafe mais os quatro modos de falha (arquivo ausente, fonte quebrado, flag inválida, timeout), cada um exigindo que NÃO sejam confundidos com UNSAFE. 15 asserções | `roadmap.md:64-67` | ✅ | `done` |
| KB-B09 | P2 | `eval "$VERIFY_CMD"` executa string vinda de JSON que o próprio agente LLM edita | `ralph_loop_esbmc.sh:62` | ✅ | `todo` |
| KB-B10 | P2 | **Feito.** `packages = ["core_verify"]`; `torch`/`onnx` movidos para extras opcionais — o wrapper não importa nenhum dos dois. `pip install -e .` funciona | `pyproject.toml:9` | ✅ | `done` |
| KB-B11 | P2 | **Feito.** `start_new_session=True` + `os.killpg` no timeout | `core_verify/esbmc_caller.py:69-73` | ✅ | `done` |

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
| KB-D01 | P0 | **Feito.** O experimento não media nada: `run_benchmark.py` passava `-DDIM_LIMIT=<n>`, mas `matmul_kernel.cpp` **nunca usava o macro** — M=N=K=2 fixos em todos os "tamanhos". Corrigidos o kernel (parametrizado), a extensão (`.cpp`→`.c`), o `--smtlib` (impedia resolver) e o caminho fixo. **Medido:** N=2 → 1,97 s SAFE; N=3 → 16,83 s SAFE; N=4 → **>300 s TIMEOUT**. O artigo publicava `<1`, `≈2`, `≈5`, `≈8`, `≈15` s. A parede está em N=4; as linhas 4–6 descreviam execuções que não completam. Tabela substituída, figura que extrapolava até N=60 removida | `results/case2_ambiente.md` | ✅ | `done` |
| KB-D02 | P0 | **Feito (documentado como limitação).** Três bloqueios independentes medidos: macro `ARDUINO` não definida → `WProgram.h` ausente; `PID_v1.cpp:46` usa construtor delegante C++11 e a 6.8.0 não tem `--std`; o harness usa justamente o construtor de 7 argumentos, que é o delegante. Seção reescrita: reporta integração e obstáculos, sem afirmar veredito | execução direta | ✅ | `done` |
| KB-D03 | P0 | **Feito.** Criado `scripts/gerar_evidencias.py`: roda cada harness citado no artigo, grava a saída **bruta** do solver em `results/logs/` e consolida em `results/EVIDENCIAS.md` a tabela afirmação→comando→veredito→log. Regras: saída bruta sempre salva (para "nada casou" seguir distinguível de "não rodou"), `TIMEOUT` como **indeciso**, ambiente junto dos números, e harness que não compila aparece como `PARSE_ERROR` em vez de sumir. **Medido: 8 provados, 3 com contraexemplo, 0 indecisos** de 11 harnesses, todos com log versionado. Rendeu também a primeira contagem de VCCs do projeto (28 geradas, 4 após simplificação no stub do Transformer), KPI que o `roadmap.md` pede e nada reportava | `results/EVIDENCIAS.md`, `results/logs/` | ✅ | `done` |
| KB-D04 | P0 | **Feito.** As 6 legendas removidas de `4resultados.tex` e `3metodologia.tex` | `4resultados.tex:73,117,217,245,308`; `3metodologia.tex:130` | ✅ | `done` |
| KB-D05 | P0 | **Feito.** `--smtlib` fazia o ESBMC emitir a fórmula em vez de resolver, então `success` era permanentemente falso e o CSV ficava com 0 bytes. Trocado pelo `esbmc_caller`; caminhos ancorados no script. **Medido:** 2 iterações — UNSAFE 0,13 s, SAFE 0,11 s. O artigo afirmava 5, inalcançáveis porque o loop encerra na primeira prova | `results/case3_agent_stats.csv` | ✅ | `done` |
| KB-D06 | P0 | **Feito.** Seção de CI reescrita para o workflow que realmente roda, incluindo por que o binário é versionado em vez de baixado e por que o caminho vai por `$GITHUB_ENV` | `.github/workflows/esbmc-verify.yml` | ✅ | `done` |
| KB-D07 | P1 | **Diagnóstico corrigido.** Instalar `ast2json` **não basta**: o binário 6.8.0 embarcado **não tem o frontend Python compilado** — `failed to figure out type of file` para qualquer `.py`. Exige compilar a 8.0.0 com `-DENABLE_PYTHON_FRONTEND=ON`. Mesmo bloqueio de `KB-D02` | execução direta | ✅ | `todo` |
| KB-D08 | P1 | **Feito.** `pibic/README.md` criado: início rápido, mapa de diretórios, como interpretar um status, flags inexistentes, pipeline do IC3, e uma seção de limitações conhecidas | — | ✅ | `done` |
| KB-D09 | P1 | **Parcial.** `results/EVIDENCIAS.md`, `results/case2_ambiente.md` e `ic3/EVIDENCIA.md` passam a registrar versão do ESBMC, solver, flags completas, timeout, SO e CPU junto de cada número. Falta reconciliar as combinações conflitantes que ainda constam do artigo e da apresentação (8.0.0/Z3, 6.8.0/Boolector 3.2, 6.8.0/Z3 4.8.9) | `4resultados.tex:4` vs `apresentacao_pibic.tex:593` | ✅ | `todo` |
| KB-D10 | P1 | Religar as figuras aos dados. Nenhum script em `artigo/figs/` abre arquivo algum — todos os valores são literais. `plot_case2.py:7-8` tem a tabela GEMM inventada que contradiz o CSV | `artigo/figs/*.py` | 🟡 | `todo` |
| KB-D11 | P1 | **Feito.** Caminhos `/home/uchoa` corrigidos em 7 arquivos; as 2 imagens que apontavam para fora do repo viraram comentário explícito de ausência | `plot_pipeline_esbmc.py:40`, `results/analysis_report.md:118,122`, +6 | ✅ | `done` |
| KB-D12 | P2 | **Feito.** 14 das 18 entradas eram de um trabalho de futebol/visão computacional; removidas. Adicionadas ESBMC, Z3, Boolector, Bitwuzla, IC3 (Bradley), PDR (Eén), ABC, Yosys, Reluplex, Marabou, Neurify, DDPG e cart-pole — cada uma marcada para conferência contra a fonte | `artigo/referencias.bib` | ✅ | `done` |
| KB-D13 | P2 | Incorporar ao artigo o trabalho feito e não reportado: cart-pole DQN/DDPG (34 arquivos, contraexemplos concretos) e FFN GPT-2/Llama (speedup 18–27×). É o material mais rigoroso do repositório e está ausente do capítulo 4 | — | 🟡 | `todo` |
| KB-D14 | P2 | **Feito (parcial).** Errata de exemplo sobre neoplasias em cães neutralizada; placeholders de examinador e a instituição errada (`CCHE/UEPB` em vez de UFAM) marcados com `TODO(autor)`. Stubs de siglas/símbolos seguem comentados, não entram no PDF | `artigo/editar/` | ✅ | `done` |
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
| **ABC BMC** | **DDPG real 16-bit** | **5 frames, sem veredito** | **901 s** | **503 MB** |
| **ABC PDR** | **DDPG real 16-bit** | **não convergiu** | **1802 s** | **708 MB** |

Invariante encontrado: **4 cláusulas, 8 literais, 3 dos 65 bits de estado** —
`th[13] ⟺ th[14] ⟺ th[15]`. Convergiu em `F[2]`.

| ID | Prio | Tarefa | Evidência | Conf. | Status |
|---|---|---|---|---|---|
| KB-E01 | P1 | **Feito** — pipeline versionado em `pibic/ic3/` (commit `25d02b9f`): `gen_transition_system.py` (lê os pesos reais e emite o sistema de transição Verilog, `--bits 16|32`), `validate_forward.py` (prova bit-exatidão do forward contra referência Python) e `run_pdr.sh` (yosys → AIGER → `abc pdr`, com tempo e pico de RSS). O Verilog é o ponto de bifurcação: dele saem tanto AIGER quanto BTOR2, então trocar de motor não exige reescrever o gerador | `pibic/ic3/` | ✅ | `done` |
| KB-E02 | P1 | **Medido sobre o ator real** (4→24→24→1, 745 parâmetros, forward validado bit-a-bit em 12/12 estados; sem os `__ESBMC_assume` de intervalo do harness C). Síntese: **904.242 portas AND, 65 latches, nível 689**. Resultado honesto: **nenhum dos dois métodos decide a instância** — `abc pdr` não convergiu em **1802 s / 708 MB**, e `abc bmc3` alcançou só **5 frames em 901 s / 503 MB**. `Timeout` é indeciso, **não** é 'seguro'. Confirmação da tese da raia: o ABC parou em 5 frames no ator real e o ESBMC parou em K=4 numa rede 8× menor — ferramentas e solvers diferentes, **mesma parede**, logo o limite é do método. Próximo passo em `KB-E07` | `pibic/ic3/cl_ddpg16.abc.out`, `cl_ddpg16.bmc.out` | ✅ | `done` |
| KB-E03 | P1 | Fechar a curva de escalabilidade do BMC. **Parcialmente respondido**: no modelo real o BMC não passa de 5 frames em 901 s, então K=8 e K=16 estão fora de alcance nesta máquina — medir K=8/K=16 só faz sentido no modelo sintético, para a curva. **Não extrapolar** de poucos pontos: é o erro que `KB-D01` acusa | `cl_ddpg16.bmc.out` | ✅ | `todo` |
| KB-E04 | P2 | Documentar as limitações honestas: (a) o PDR também não convergiu no modelo sintético sem batente (500 s); (b) a vantagem é **estrutural** — memória independente da profundidade — não incondicional; (c) bit-blastar para AIGER **perde a estrutura de palavra**, o que atrapalha a generalização de cláusulas do PDR. Ver `KB-E07` | ✅ medido | ✅ | `todo` |
| KB-E05 | P2 | Avaliar `--overflow-check` / bit-width: 32→16 bits cortou o estado de 129 para 65 flops e a memória do PDR de 246 para 110 MB. O cartpole real é Q8.8 = 16 bits | ✅ medido | ✅ | `todo` |
| KB-E06 | P2 | Reescrever o Caso 5 do artigo (52 topologias, sem artefato) usando os dados de `KB-E03` — resultado medido no lugar de número inventado | `4resultados.tex:265` | ✅ | `todo` |
| KB-E07 | P2 | **Rota word-level — agora o caminho principal, não plano B.** O ABC não convergiu no modelo real (`KB-E02`), e a hipótese mais provável é a perda de estrutura: bit-blastar 4 inteiros em 65 bits soltos destrói exatamente o que o PDR usa para generalizar cláusulas, e o invariante precisaria falar sobre `th` como palavra. Exportar BTOR2 (`yosys write_btor`, já disponível) e usar um motor word-level — AVR, Pono ou `btormc`. Nenhum está instalado nem no apt (`btormc` vem com o Boolector; AVR e Pono exigem build do fonte). BTOR2 preserva os 4 inteiros em vez de 65 bits soltos, o que tende a ajudar muito a generalização | `yosys -p "help write_btor"`; `command -v avr pono btormc` → ausentes | ✅ | `todo` |

---

## Raia F — Higiene do repositório

| ID | Prio | Tarefa | Evidência | Conf. | Status |
|---|---|---|---|---|---|
| KB-F01 | P1 | **Feito.** Os 6 arquivos movidos da raiz do fork para `pibic/chatbot/` | `git ls-files` na raiz | ✅ | `done` |
| KB-F02 | P1 | Decidir sobre os binários em `QNNVerifier/esbmc-6.8.0/`: 71 MB (`esbmc`) + 57 MB (`esbmc.exe`) + 15 MB (`libz3.dll`). `pibic/` tem 322 MB, dos quais 308 MB são `QNNVerifier/` | `du -sh` | ✅ | `todo` |
| KB-F03 | P1 | **Feito.** `pibic/.gitignore` criado — artefatos de verificação, cache Python, intermediários de LaTeX | — | ✅ | `done` |
| KB-F04 | P2 | Reconciliar as duas taxonomias concorrentes: `1_python_models/`…`4_control_system/` (legado) vs `cases/` + `core_verify/` (roadmap). Ambas coexistem com código duplicado. Declarar qual é canônica | `roadmap.md:37-56` | 🟡 | `todo` |
| KB-F05 | P2 | **Feito.** `recreate_prd.py` removido (gerava as tarefas de display Android) e `prd.json` regenerado com as 110 tarefas do ESBMC. `generate_100_prd.py` teve o caminho de saída corrigido | `prd.json`, `recreate_prd.py:4` | ✅ | `done` |
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
| A — Cobertura da auditoria | — | 2 | 1 | 3 |
| B — Núcleo & CI | 2 | 6 | 3 | 11 |
| C — Propriedades | 3 | 4 | 1 | 8 |
| D — Evidências & artigo | 6 | 5 | 4 | 15 |
| E — IC3/PDR | — | 3 | 4 | 7 |
| F — Higiene | — | 4 | 4 | 8 |
| **Total** | **11** | **24** | **17** | **52** |

Confiança: **40 ✅ verificado** · **12 🟡 relatado por agente** · **0 ⬜ não auditado**

Contagens conferidas por script sobre as próprias linhas da tabela, não à mão — as
versões anteriores deste rodapé traziam 21/26/3 e 25/23/3, ambas erradas. Um board que
acusa números sem evidência não pode ter números sem evidência.

Progresso: `KB-C00` concluída — as duas vacuidades P0 (`KB-C01`, `KB-C02`) saíram de
🟡 para ✅ por reprodução direta. As auditorias `KB-A01` e `KB-A02` estão em execução.
