# Plano do Relatório Final PIBITI 2025/2026

Plano elaborado a partir do modelo oficial atual da PROTEC, da estrutura dos trabalhos premiados no CONIC/UFAM entre 2021 e 2025 e da auditoria dos arquivos existentes no projeto.

## Decisão narrativa recomendada

Manter como título oficial, salvo divergência no eCampus ou no termo de concessão:

> **Verificação Formal de Sistemas de Inteligência Artificial Generativa e Controladores Neurais utilizando ESBMC**

O relatório deve contar uma história principal, e não seis histórias concorrentes. A formulação recomendada é:

> Investigar até que ponto a verificação baseada em ESBMC consegue produzir garantias úteis, contraexemplos reproduzíveis e diagnósticos de limite para componentes neurais quantizados e controladores, sob hipóteses explicitamente definidas.

O estudo CartPole deve ser o **caso principal**, porque hoje é o conjunto mais coerente de método, números, artefato, prova, refutação e limitação. Casos ligados a GenAI, kernels ou agentes só devem permanecer como validações complementares se seus resultados puderem ser reexecutados e rastreados até logs/CSV reais. Caso contrário, devem ir para “trabalhos em andamento”, limitações ou ser removidos.

## Perguntas de pesquisa

- **PQ1 — Fidelidade:** a representação Q8.8 preserva adequadamente o comportamento do controlador Float32 no domínio operacional definido?
- **PQ2 — Estrutura neural:** existem neurônios mortos ou permanentemente saturados no domínio analisado?
- **PQ3 — Segurança:** quais propriedades de malha fechada podem ser provadas ou refutadas pelo ESBMC e que informação prática os contraexemplos fornecem?
- **PQ4 — Limites:** quais hipóteses, custos e timeouts limitam a conclusão formal?
- **PQ5 — Generalização metodológica:** quais partes do fluxo podem ser reaproveitadas na verificação de componentes de IA generativa? Esta pergunta deve ser respondida apenas com casos realmente executados.

## Estrutura alinhada ao modelo oficial PROTEC

### 1. Identificação

Copiar exatamente do eCampus/termo: título, programa, modalidade, período, bolsista, orientador, unidade e projeto. Não alterar o título para tornar o texto mais atraente sem confirmar o vínculo oficial.

### 2. Resumo

Escrever por último, em um único bloco curto com seis movimentos:

1. contexto e risco de controladores neurais;
2. lacuna: testes não cobrem todo o domínio e a quantização pode separar o modelo verificado do executado;
3. objetivo;
4. método: ESBMC, harnesses C, Q8.8, domínio e simulador;
5. resultados numéricos auditados;
6. conclusão e principal limitação.

O resumo deve mencionar apenas números já ligados a arquivos brutos. A formulação atual pode aproveitar: 48 neurônios, 10.000 estados, propriedade de força provada, segurança em um passo refutada e correção direcional com timeout — desde que todos sejam reexecutados e arquivados antes da submissão.

### 3. Introdução

Sequência recomendada:

1. redes neurais em sistemas ciberfísicos produzem decisões difíceis de inspecionar;
2. testes amostrais são úteis, mas não equivalem a uma análise de todo o domínio;
3. BMC/SMT e ESBMC permitem formular propriedades sobre implementações C/C++;
4. quantização oferece eficiência, mas cria uma obrigação de fidelidade;
5. lacuna concreta: integrar o controlador quantizado, a verificação e a reprodução do contraexemplo;
6. pergunta central e contribuições;
7. mapa curto do restante do relatório.

Evitar promessas amplas como “garantir a segurança da IA generativa”. Preferir: “provar ou refutar propriedades específicas, dentro das hipóteses e limites declarados”.

### 4. Objetivos

**Objetivo geral sugerido**

> Desenvolver e avaliar um fluxo reprodutível de verificação formal com ESBMC para propriedades estruturais e de segurança de componentes neurais quantizados e controladores, produzindo provas, contraexemplos e diagnósticos de limitações sob domínios explícitos.

**Objetivos específicos sugeridos**

1. converter o controlador neural para uma representação Q8.8 compatível com a aritmética executada;
2. medir a diferença entre as saídas Float32 e Q8.8 em uma amostra definida;
3. formular harnesses para atividade/saturação neural e propriedades de controle;
4. executar e registrar o ESBMC com versão, solver, flags e timeout fixados;
5. reproduzir contraexemplos no simulador;
6. caracterizar as limitações de escalabilidade e validade;
7. avaliar a reutilização do fluxo em um componente de IA generativa, somente se houver evidência executável.

Cada objetivo deve ter ao menos uma linha correspondente na tabela final de resultados.

### 5. Metodologia

#### 5.1 Objeto de estudo

- ambiente CartPole e variáveis de estado;
- controlador DDPG 4–24–24–1;
- ativações ReLU/Tanh e faixa de força;
- origem dos pesos e identificação do checkpoint;
- domínio operacional e domínio seguro, com unidades.

#### 5.2 Quantização e fidelidade

- definição formal de Q8.8 e fator 256;
- regra de arredondamento e divisão com truncamento no estilo C;
- aproximação por partes da Tanh;
- justificativa de usar a mesma implementação nos harnesses e no simulador;
- desenho da comparação Float32 × Q8.8: distribuição dos 10.000 estados, semente e métricas.

#### 5.3 Propriedades verificadas

Definir cada propriedade antes de apresentar resultados:

| Código | Propriedade | Domínio | Resultado possível |
|---|---|---|---|
| P1 | atividade de cada neurônio | entradas operacionais | ativo/morto |
| P2 | saturação permanente | entradas operacionais | saturado/não saturado |
| P3 | correção direcional | região de perigo | provada/refutada/timeout |
| P4 | segurança em um passo | estado inicialmente seguro | provada/refutada/timeout |
| P5 | limite de força | domínio operacional | provada/refutada/timeout |

Explicar com precisão a semântica de `assume`, `assert`, `VERIFICATION SUCCESSFUL`, `VERIFICATION FAILED` e timeout. “FAILED” só é evidência da violação da asserção especificada, não de uma falha global do sistema.

#### 5.4 Configuração experimental e reprodutibilidade

Registrar:

- versão/commit do ESBMC;
- solver e versão;
- sistema operacional, CPU e memória;
- comando exato e flags;
- timeout;
- scripts geradores dos harnesses;
- hash ou caminho dos pesos;
- sementes aleatórias;
- estrutura dos arquivos de saída;
- procedimento para injetar um contraexemplo no simulador.

#### 5.5 Casos complementares

Usar uma subseção curta por caso, sempre com a mesma moldura: objetivo, artefato, propriedade, comando, resultado e limitação. Um caso que não tenha os seis itens não deve entrar como resultado concluído.

### 6. Resultados e discussão

Organizar por pergunta de pesquisa, não por ordem cronológica de implementação.

#### 6.1 Fidelidade Float32 × Q8.8

- tabela com tamanho da amostra, erro médio absoluto, erro máximo e percentis;
- figura de distribuição do erro ou dispersão Float32 × Q8.8;
- interpretação prática do erro no comando de força;
- limitação: amostragem não é prova para todo o domínio.

#### 6.2 Estrutura da rede

- tabela por camada, com total de neurônios, ativos, mortos e saturados;
- manter a conclusão “48 ativos” apenas se os 48 logs individuais ou um arquivo consolidado forem gerados;
- explicar corretamente por que um contraexemplo à hipótese “sempre zero” demonstra atividade.

#### 6.3 Propriedades do controlador

Tabela central recomendada:

| Propriedade | Resultado ESBMC | Tempo | Evidência | Interpretação |
|---|---|---:|---|---|
| correção direcional | timeout | a medir | log bruto | inconclusiva no orçamento usado |
| segurança em um passo | refutada | a medir | log + estado inicial | revela limite do domínio/invariante |
| limite de força | provada | a medir | log bruto | garantia no domínio declarado |

O contraexemplo de segurança deve receber uma figura própria: estado inicial, ação, estado seguinte, limite violado e reprodução no simulador. Essa é uma contribuição central, não uma nota de rodapé.

#### 6.4 Discussão

Comparar três tipos de evidência:

- testes amostrais: rápidos, mas incompletos;
- prova formal: conclusiva apenas para a propriedade, modelo e domínio definidos;
- contraexemplo: diagnóstico concreto que permite refinar o requisito.

Relacionar os achados à literatura de BMC, ESBMC, redes quantizadas e verificação de controladores. Declarar ameaças à validade: dinâmica linearizada, um único passo, aproximação Tanh, discretização Q8.8, checkpoint específico e orçamento de timeout.

### 7. Conclusão

Responder às perguntas em ordem:

- fidelidade observada da quantização;
- ausência/presença de neurônios mortos ou saturados;
- propriedades provadas;
- propriedades refutadas e o que o contraexemplo ensinou;
- propriedades inconclusivas por timeout;
- alcance e limites da contribuição;
- um ou dois próximos passos concretos.

Não introduzir números novos nem afirmar que timeout confirma segurança ou insegurança.

### 8. Referências

Priorizar fontes primárias: artigos originais de BMC/ESBMC, documentação ou artigo do solver, trabalho de quantização/verificação mais próximo, DDPG e referências de controle/CartPole. Remover referências herdadas do template e confirmar que toda entrada foi citada no texto.

### 9. Termo/anexo obrigatório

Preservar o termo existente no modelo oficial PROTEC e conferir assinatura, identificação e ordem de páginas. Anexos técnicos só devem ser adicionados se o limite de 2 MB permitir e se o modelo aceitar; comandos e dados extensos podem ser indicados por repositório persistente.

## Matriz afirmação–evidência

Esta matriz deve ser preenchida antes da redação final:

| Afirmação pretendida | Evidência mínima | Estado atual |
|---|---|---|
| Q8.8 é fiel na amostra | CSV dos 10.000 estados + script + métricas | confirmar/reexecutar |
| 48 neurônios estão ativos | logs por neurônio ou JSON consolidado | confirmar/reexecutar |
| nenhum neurônio satura permanentemente | logs e definição por camada | confirmar/reexecutar |
| limite de força foi provado | comando, versão e log `SUCCESSFUL` | confirmar/reexecutar |
| segurança em um passo foi refutada | log `FAILED`, valores do contraexemplo e reprodução | confirmar/reexecutar |
| correção direcional deu timeout | dois logs e timeout configurado | confirmar/reexecutar |
| caso Python/MLP foi verificado | ambiente com `ast2json` e log bem-sucedido | atualmente contradito por log |
| benchmark GEMM escala de N=2 a N=6 | CSV real para todos os N | atualmente há somente N=2 e N=3 |
| agente realizou cinco iterações | log/CSV não vazio e prompts/saídas | atualmente sem evidência |
| 52 casos foram avaliados | dataset consolidado com 52 linhas | atualmente sem evidência bruta |

As quatro últimas afirmações não devem permanecer na versão submetida sem nova execução. Gráficos produzidos por vetores manuais, tempos simulados ou dados sintéticos não podem ser apresentados como resultado experimental.

## Figuras e tabelas prioritárias

1. pipeline “modelo treinado → quantização → harness C → ESBMC → prova/contraexemplo → simulador”;
2. domínio de estado e propriedades verificadas;
3. erro Float32 × Q8.8 com dados reais;
4. resumo estrutural dos 48 neurônios;
5. contraexemplo de segurança em um passo;
6. tabela central de propriedades e resultados;
7. tabela de configuração reprodutível.

Cada figura deve ter uma mensagem científica única, legenda autossuficiente, e fonte dos dados. Excluir decoração, superfícies geradas apenas para aparência e qualquer curva simulada que não corresponda ao experimento descrito.

## Portões de qualidade antes da submissão

- [ ] modalidade, título e período confirmados no eCampus/termo;
- [ ] modelo oficial PROTEC usado como autoridade de estrutura;
- [ ] todas as afirmações centrais ligadas a evidência bruta;
- [ ] resultados sintéticos ou contraditórios removidos;
- [ ] comandos, versões, solver, hardware e timeout documentados;
- [ ] figuras regeneradas exclusivamente de CSV/log real;
- [ ] resumo e conclusão numericamente consistentes com os resultados;
- [ ] documento não contém “Resultado Parcial”, UEPB, CCHE ou texto legado de TCC;
- [ ] PDF recompilado sem erros, fontes incorporadas e páginas conferidas visualmente;
- [ ] arquivo final com no máximo 2 MB;
- [ ] envio efetuado com antecedência e comprovante salvo.

## Ordem de execução em 30–31 de agosto

1. **Travar a verdade administrativa:** confirmar PIBITI, título e dados no eCampus.
2. **Travar a verdade experimental:** reexecutar o núcleo CartPole e arquivar logs/CSV.
3. **Cortar o que não é comprovável:** remover casos e números sem rastreabilidade.
4. **Reescrever em torno das perguntas:** objetivos, método, resultados, discussão e conclusão.
5. **Gerar figuras dos dados reais:** somente após os CSV/logs finais.
6. **Conformidade:** migrar/ajustar para a estrutura PROTEC e preservar o termo.
7. **QA final:** compilar, revisar visualmente, conferir 2 MB e submeter antes do fim do dia.

As fontes oficiais consultadas indicam a data de 31 de agosto de 2026, mas não publicam horário-limite. Portanto, o plano não deve depender de uma suposição de 23h59.
