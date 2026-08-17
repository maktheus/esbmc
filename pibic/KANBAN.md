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
   registrados. É o que separa os 37 ✅ dos 13 🟡.
3. **Medição, não estimativa**, onde havia número em jogo: a parede do BMC e o
   resultado do IC3 na raia E vêm de execução cronometrada, não de extrapolação.

### Estado agora

**26 das 52 tarefas concluídas**, entre elas **10 dos 11 P0**. A suíte saiu de
*21 falhas em 0,12 s* — porque nada rodava — para **31 testes passando em 12m49s**,
verificando de verdade.

| | |
|---|---|
| **Núcleo consertado** | O wrapper distingue erro de execução de propriedade violada; a CI passou a existir de fato. |
| **Cinco vacuidades confirmadas** | Properties B e C do cartpole; `kernels_benchmarks.cpp` que não compilava (15 testes inertes); o ruído dos 5 perfis de caos que nunca chegava ao sistema; e o `DIM_LIMIT` do GEMM que nunca era usado. |
| **Artigo alinhado à medição** | Tabela do GEMM substituída por dado real (a parede está em N=4, não em N=60); seção da Arduino PID reescrita como limitação; CI descrita como o workflow que roda; bibliografia reconstruída. |
| **Raia E encerrada** | Empate técnico no ator real: nenhum método decide. |
| **Em aberto** | As duas auditorias perdidas (`KB-A01`, `KB-A02`), o P0 restante (`KB-D03`, logs de execução), e itens P1/P2 de documentação. |

