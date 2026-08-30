# Plano de execução detalhado — Relatório Final PIBITI 2025/2026

Este documento descreve como transformar o material atual em um relatório final tecnicamente defensável, rastreável e competitivo. Ele complementa `PLANO_RELATORIO_FINAL_PREMIAVEL.md`: aquele documento define a arquitetura científica; este define a execução, os arquivos, as verificações e os critérios de aceite.

## 1. Resultado final esperado

Ao terminar, o projeto deverá conter:

1. um relatório final com a estrutura exigida pela PROTEC;
2. uma narrativa central baseada no controlador DDPG do CartPole;
3. evidências brutas imutáveis para todos os números apresentados;
4. figuras e tabelas geradas exclusivamente dessas evidências;
5. um pacote mínimo de reprodução com ambiente, comandos e hashes;
6. um PDF visualmente revisado, sem erros e com no máximo 2 MB;
7. uma cópia do arquivo submetido e do comprovante do eCampus.

O título de trabalho será **“Verificação Formal de Sistemas de Inteligência Artificial Generativa e Controladores Neurais utilizando ESBMC”**, mas deverá ser comparado literalmente ao cadastro do eCampus antes da montagem da capa.

## 2. Decisões de escopo

### 2.1 Eixo principal

O estudo CartPole/DDPG será responsável pela maior parte da metodologia e dos resultados. A contribuição será apresentada como um fluxo integrado:

> checkpoint DDPG → quantização Q8.8 → harnesses C → ESBMC → prova/contraexemplo/timeout → reprodução no simulador.

Essa escolha permite responder perguntas específicas sobre fidelidade, atividade dos neurônios e segurança do controlador.

### 2.2 Casos complementares

Os casos de IA generativa serão tratados como extensão do fluxo, não como quatro ou seis contribuições equivalentes. Cada caso só permanece se possuir:

- código executável;
- propriedade formal explicitada;
- comando registrado;
- log bruto;
- resultado interpretável;
- limitação declarada.

O caso Python/MLP, o benchmark GEMM, o agente neuro-simbólico e a alegação de 52 avaliações estão inicialmente em **quarentena editorial**. Eles não serão usados como resultados até satisfazerem os critérios acima.

### 2.3 O que não será afirmado

- que o sistema de IA é “totalmente seguro”;
- que timeout prova segurança ou insegurança;
- que equivalência de implementação significa erro numérico zero entre Float32 e Q8.8;
- que dados sintéticos ou tempos simulados são resultados experimentais;
- que um contraexemplo a uma propriedade específica caracteriza falha global do controlador.

## 3. Organização proposta dos arquivos

O relatório final será montado em uma pasta nova, preservando `pibic/artigo/` como histórico:

```text
pibic/relatorio_final_2026/
├── main.tex
├── config/
│   └── dados.tex
├── secoes/
│   ├── 00_resumo.tex
│   ├── 01_introducao.tex
│   ├── 02_objetivos.tex
│   ├── 03_metodologia.tex
│   ├── 04_resultados_discussao.tex
│   └── 05_conclusao.tex
├── figuras/
│   ├── fonte/
│   └── geradas/
├── tabelas/
│   └── geradas/
├── anexos/
├── referencias.bib
└── build.ps1

pibic/evidencias/final_2026/
├── manifesto.json
├── ambiente/
├── quantizacao/
├── neuronios/
├── malha_fechada/
├── contraexemplos/
└── casos_complementares/
```

O diretório `evidencias/final_2026` será somente acrescentado durante a rodada final: novas execuções não deverão apagar logs anteriores. O `manifesto.json` registrará data, commit, sistema, CPU, memória, Python, ESBMC, solver, checkpoint, hashes e comandos.

## 4. Fase 0 — confirmação administrativa

### Ações

1. abrir o cadastro no eCampus;
2. confirmar se a modalidade é PIBITI;
3. copiar literalmente título, bolsista, orientador, unidade e vigência;
4. copiar o código do projeto e a grande área CNPq;
5. registrar a modalidade de bolsa (CNPq, UFAM ou voluntário);
6. marcar possibilidade de patente/registro de software e necessidade de apresentação reservada;
7. confirmar o apoio financeiro/institucional que deve aparecer nos agradecimentos;
8. confirmar quem possui o perfil responsável pelo envio;
9. verificar o Termo de Autorização e Declaração de Distribuição Não Exclusiva para o RIU;
10. registrar uma captura ou exportação desses dados fora do PDF final.

### Dependência do usuário/orientador

O usuário deverá confirmar os dados que não existem de forma confiável no repositório. Nenhuma capa será considerada final antes disso.

### Critério de aceite

Todos os campos da capa, as declarações de propriedade intelectual/sigilo e o termo do RIU coincidem com o eCampus. Não existem referências a PIBIC, UEPB ou CCHE em campos administrativos.

## 5. Fase 1 — congelamento e auditoria da evidência

### 5.1 Identificar o artefato canônico

Será usado o checkpoint:

`pibic/cartpole/ddpg_actor_best.pth`

Antes da execução, serão calculados hashes SHA-256 do checkpoint, dos scripts e dos pesos exportados. Isso evita comparar resultados produzidos por versões diferentes sem perceber.

### 5.2 Registrar o ambiente

O binário existente foi identificado como **ESBMC 6.8.0, Linux x86_64**, executável pelo WSL. Serão registrados:

- distribuição e versão do WSL;
- Python e dependências;
- ESBMC 6.8.0;
- solver selecionado (`--boolector`);
- CPU, memória e sistema;
- timeout por propriedade.

### 5.3 Resolver divergências atuais

Antes de escrever resultados, quatro conflitos deverão ser fechados:

| Conflito | Evidência atual | Ação |
|---|---|---|
| correção direcional | o texto diz dois timeouts; `ddpg_closed_loop_results.json` registra dois `FAILED` com contraexemplos | executar novamente com um orçamento único, salvar stdout e validar cada contraexemplo |
| fidelidade Q8.8 | média 0,0778 N, mas máximo 7,7589 N e p99 1,0927 N | salvar as 10.000 amostras, localizar o pior caso e medir troca de sinal/ação |
| identidade do controlador | há resultados DQN e DDPG em arquivos parecidos | separar completamente resultados por controlador e usar somente DDPG no eixo principal |
| tempo de execução | JSON antigo contém 1.888 s e 2.722 s, embora o script atual declare timeout de 120 s | registrar tempo automaticamente na nova rodada e não reutilizar tempos antigos sem proveniência |

Há ainda um portão de validade formal que precisa ser resolvido antes de qualquer afirmação sobre a rede completa:

- o script de neurônios restringe pré-ativações internas a `[-2048,2048]` na camada 1 e `[-4096,4096]` na camada 2, mas os próprios pesos e bounds declarados chegam aproximadamente a `[-5690,4150]` e `[-60954,36419]`;
- na camada 2, `h1` e `h2` são tratados como variáveis simbólicas independentes dentro de intervalos, em vez de a segunda camada ser calculada a partir da primeira no mesmo harness;
- por isso, os resultados atuais podem ser provas sobre uma sobreaproximação ou sobre um subdomínio, e não provas sobre todos os estados da rede real;
- o mesmo cuidado se aplica à responsividade da saída;
- a alegação textual de erro inferior a 3% para a aproximação da Tanh também precisa ser recalculada: a implementação atual apresenta erro absoluto máximo aproximado de 4,57% em uma avaliação independente.

**Regra:** até que esses pontos sejam corrigidos, o relatório deve chamar os resultados de preliminares/análises sobre o harness atual, nunca de prova da rede completa.

### Critério de aceite

Existe uma única rodada canônica identificada por data/hash, e o texto pode ser reconstruído usando somente os arquivos dessa rodada.

## 6. Fase 2 — reexecução dos experimentos CartPole

Será criado um `run_final_evidence.py` ou equivalente para executar o fluxo e armazenar cada comando, harness e stdout em uma pasta própria. A execução manual abaixo serve como referência, mas o runner final deverá impedir sobrescrita silenciosa:

```bash
cd /mnt/e/Uchoa/pibic/pibic/cartpole
python3 export_quantized_weights.py
python3 verify_ddpg_dead_neurons.py --all
python3 verify_ddpg_saturation.py
python3 verify_ddpg_closed_loop.py
python3 generate_ddpg_webapp_data.py
```

### 6.1 Quantização Q8.8

O script será ajustado para salvar, para cada uma das 10.000 amostras:

- índice e semente;
- quatro variáveis do estado;
- saída Float32;
- saída Q8.8;
- erro absoluto e assinado;
- sinal de cada saída;
- indicador de discordância de direção;
- distância do estado aos limites do domínio.

Serão calculados média, mediana, desvio, máximo, p90, p95 e p99. O pior caso de 7,7589 N deverá ser reproduzido e explicado. Também será medida a taxa de discordância de sinal, pois um erro pequeno próximo de zero pode mudar a direção da força.

**Figura:** dispersão Float32 × Q8.8 com linha ideal e destaque dos maiores erros.

**Tabela:** métricas de erro e discordância de sinal.

**Critério de aceite:** CSV com exatamente 10.000 linhas, semente registrada, métricas recalculáveis e pior caso identificado. A expressão “fidelidade total” só poderá se referir ao uso da mesma aritmética Q8.8 no C e no TypeScript, nunca à equivalência entre Float32 e Q8.8.

### 6.2 Neurônios mortos

Serão executadas as duas camadas ocultas, 24 neurônios por camada. Primeiro serão corrigidos os bounds internos para não excluir valores possíveis. Em seguida, a camada 2 será calculada a partir da saída concreta da camada 1 no mesmo harness; não serão usados vetores intermediários independentes como substituto do grafo da rede. Para cada neurônio, serão preservados:

- harness C gerado;
- comando ESBMC;
- stdout/stderr;
- estado do resultado;
- contraexemplo que ativa o neurônio, quando encontrado;
- tempo de execução.

**Tabela:** camada, total, ativos, mortos, timeout e tempo total.

**Critério de aceite:** 48 registros individuais; nenhum resultado inferido apenas de um JSON agregado. A interpretação deve respeitar a propriedade: `FAILED` na asserção “sempre zero” fornece uma entrada que demonstra que o neurônio está ativo.

Se a versão corrigida não puder ser concluída dentro do prazo, a tabela será apresentada como análise preliminar do harness, com o subdomínio e a sobreaproximação explicitamente descritos; a frase “todos os 48 neurônios estão formalmente ativos” será retirada.

### 6.3 Saturação e responsividade

Será documentado exatamente o que o script verifica. A camada 2 e a saída deverão receber valores derivados da rede real, não variáveis intermediárias independentes. Se a camada 2 não estiver coberta, o texto não poderá afirmar ausência de saturação em toda a rede. A saída deverá ser testada para produção de forças positivas e negativas no domínio.

**Critério de aceite:** cobertura por camada explicitada e logs correspondentes. “Responsivo” significará apenas que existem entradas que produzem sinais opostos, não que a política é correta em todos os estados.

### 6.3.1 Métrica da aproximação Tanh

A aproximação por partes deverá ser avaliada com erro absoluto e, separadamente, erro relativo apenas fora de uma faixa próxima de zero. O texto não usará um percentual único sem definir denominador e domínio. O resultado atual de aproximadamente 4,57% de erro absoluto máximo será confirmado ou substituído por uma métrica regenerada.

### 6.4 Propriedades de malha fechada

As propriedades serão executadas separadamente com o mesmo orçamento declarado:

- P3-direita: região de perigo positiva implica direção esperada da força;
- P3-esquerda: região de perigo negativa implica direção esperada da força;
- P4: segurança angular após um passo;
- P5: limite de força de ±10 N.

Para cada uma serão guardados harness, limites, comando, resultado, tempo e contraexemplo. O relatório usará três estados sem ambiguidade:

- **provada no domínio:** `VERIFICATION SUCCESSFUL`;
- **refutada:** `VERIFICATION FAILED` com contraexemplo válido;
- **inconclusiva no orçamento:** timeout ou resultado desconhecido.

**Critério de aceite:** toda propriedade possui um log bruto. Contraexemplos são reavaliados pelo controlador Q8.8 fora do ESBMC e comparados ao simulador. Caso a dinâmica não linear discorde da dinâmica linearizada do harness, ambas as respostas serão apresentadas e a diferença será discutida.

### 6.5 Reprodução de contraexemplos

Será criado um pequeno arquivo de casos em JSON com estado inicial, força, estado seguinte esperado, propriedade violada e origem do log. O simulador web deverá carregar esse arquivo sem digitação manual.

**Figura:** quadro em três etapas — estado inicial, comando do controlador e estado seguinte/limite.

**Critério de aceite:** o valor reproduzido coincide com o harness Q8.8; qualquer diferença de arredondamento é registrada.

## 7. Fase 3 — decisão sobre os casos de IA generativa

Depois que o núcleo CartPole estiver fechado, será reservado um tempo limitado para um caso complementar. A ordem de preferência será:

1. caso com código simples, propriedade explícita e execução curta;
2. caso MLP/transformer apenas se `ast2json` e o frontend funcionarem;
3. kernel GEMM somente nos tamanhos realmente medidos;
4. agente neuro-simbólico somente se houver chamadas, prompts, respostas e logs reais.

### Regra de corte

Se um caso não puder ser reproduzido dentro do orçamento da rodada final, ele será descrito como protótipo/trabalho em andamento em poucas linhas ou removido. Não haverá geração artificial de pontos para completar gráficos.

### Critério de aceite

No máximo um caso complementar bem sustentado. Ele deverá demonstrar que a infraestrutura pode ser reaproveitada em GenAI sem desviar o eixo científico do relatório.

## 8. Fase 4 — redação seção por seção

### 8.1 Resumo

**Será feito por último.** Terá: contexto, lacuna, objetivo, método, três ou quatro resultados auditados, conclusão e limitação. Nenhum número entrará antes de a matriz de evidências estar verde.

### 8.2 Introdução

Cada parágrafo terá uma função:

1. risco e importância de redes neurais em controle;
2. limite dos testes amostrais;
3. oportunidade de BMC/SMT e ESBMC;
4. dificuldade da aritmética em ponto flutuante e papel da quantização;
5. lacuna: ligar artefato verificado, artefato executado e reprodução;
6. pergunta central;
7. contribuições reais;
8. organização do relatório.

**Aceite:** a introdução termina com perguntas que a conclusão consegue responder.

### 8.3 Objetivos

Um objetivo geral e objetivos específicos mensuráveis. A tabela interna de controle ligará cada objetivo a uma seção de método e a pelo menos um resultado.

**Aceite:** não existe objetivo sem resultado nem resultado importante sem objetivo associado.

### 8.4 Metodologia

Ordem:

1. visão geral do fluxo;
2. ambiente CartPole e domínio;
3. controlador/checkpoint;
4. quantização e Tanh aproximada;
5. formulação das propriedades;
6. ESBMC, solver, comandos e timeout;
7. protocolo de amostragem;
8. reprodução dos contraexemplos;
9. caso complementar, se aprovado.

**Aceite:** outra pessoa consegue entender o que repetir, com qual entrada e qual critério de saída.

### 8.5 Resultados e discussão

Ordem:

1. fidelidade Float32 × Q8.8;
2. atividade/saturação;
3. propriedades em malha fechada;
4. contraexemplo reproduzido;
5. custo/timeout;
6. caso complementar;
7. comparação com a literatura;
8. ameaças à validade.

Cada subseção seguirá: pergunta → resultado → evidência → interpretação → limitação.

**Aceite:** toda tabela/figura possui fonte de dados; a discussão distingue prova, refutação e amostragem.

### 8.6 Conclusão

Responderá em sequência:

- qual fidelidade foi observada;
- o que foi descoberto sobre os neurônios;
- quais propriedades foram provadas;
- quais foram refutadas e o que os contraexemplos revelam;
- quais ficaram inconclusivas;
- qual contribuição é reutilizável;
- quais são os próximos passos.

**Aceite:** nenhum resultado novo, nenhum adjetivo sem evidência e nenhuma generalização além do domínio.

### 8.7 Referências

As referências serão revisadas em três passagens: fonte primária, correspondência citação–bibliografia e formatação. Entrarão os trabalhos fundamentais de BMC, ESBMC, DDPG, verificação de redes neurais quantizadas e controladores.

**Aceite:** nenhuma referência herdada sem citação; nenhuma afirmação de estado da arte baseada apenas em fonte secundária.

## 9. Fase 5 — figuras e tabelas

### Conjunto mínimo

1. pipeline completo do estudo;
2. arquitetura 4–24–24–1 e domínios;
3. dispersão/erro Float32 × Q8.8;
4. resumo dos 48 neurônios;
5. tabela das propriedades formais;
6. visual do contraexemplo;
7. configuração de reprodução.

### Padrão visual

- tipografia legível no tamanho final;
- paleta pequena e compatível com impressão;
- resultado positivo, negativo e inconclusivo distinguíveis também sem cor;
- unidades nos eixos;
- legenda autossuficiente;
- saída preferencialmente vetorial em PDF;
- PNG apenas para capturas, comprimido de maneira controlada.

### Critério de aceite

Cada figura responde uma pergunta do texto e pode ser regenerada por um script ligado a CSV/JSON real.

## 10. Fase 6 — montagem LaTeX e conformidade PROTEC

O novo fonte reproduzirá a sequência de campos do modelo oficial da PROTEC. O modelo público do Overleaf será usado apenas como referência de organização visual. O estilo UEPB/CCHE atual não será carregado no novo documento, eliminando os resíduos institucionais e o erro TikZ existente.

O `build.ps1` deverá:

1. limpar somente auxiliares da pasta de build explicitamente definida;
2. executar LaTeX/BibTeX na ordem necessária;
3. falhar se houver referências indefinidas ou erro de compilação;
4. copiar o PDF final para `dist/`;
5. mostrar tamanho, número de páginas e hash SHA-256.

### Critério de aceite

- compilação limpa;
- capa e campos de acordo com a PROTEC;
- termo/anexo preservado;
- sumário e referências corretos;
- nenhuma ocorrência de “Resultado Parcial”, UEPB ou CCHE;
- PDF com no máximo 2 MB.

## 11. Fase 7 — revisão científica e visual

### Revisão científica

- conferir cada número contra a matriz de evidências;
- reler as definições formais das propriedades;
- validar semântica de `FAILED` e `SUCCESSFUL`;
- conferir domínio, unidades e arredondamentos;
- identificar promessas maiores que os resultados.

### Revisão textual

- remover linguagem promocional;
- reduzir períodos longos;
- padronizar “contraexemplo”, ESBMC, Q8.8, Float32 e CartPole;
- conferir português, siglas e referências cruzadas;
- garantir que resumo e conclusão concordam.

### Revisão visual

Renderizar todas as páginas e inspecionar capa, quebras, tabelas, figuras, legendas, cabeçalhos, páginas vazias, referências e termo.

### Revisão do orientador

Enviar uma versão fechada com uma lista curta de decisões pendentes, evitando solicitar revisão de conteúdo que ainda está mudando.

## 12. Fase 8 — pacote e submissão

1. gerar `Relatorio_Final_PIBITI_2025_2026.pdf`;
2. confirmar tamanho menor ou igual a 2 MB;
3. abrir o arquivo final e verificar páginas inicial/final e figuras;
4. calcular hash SHA-256;
5. enviar pelo perfil responsável no eCampus;
6. salvar comprovante/captura com data e hora;
7. preservar exatamente o PDF enviado.

As fontes oficiais informam 31 de agosto de 2026, mas não um horário. A meta operacional será submeter durante o dia, com margem para correção.

## 13. Cronograma de prioridade

### 30 de agosto — prioridade absoluta

- confirmar dados administrativos;
- criar a estrutura de evidências;
- ajustar o runner para não sobrescrever resultados;
- executar quantização e localizar o erro máximo;
- iniciar as verificações curtas de neurônios e saturação;
- definir a rodada canônica de malha fechada.

### 31 de agosto — manhã

- fechar resultados e contraexemplos;
- cortar casos sem evidência;
- gerar tabelas e figuras;
- escrever metodologia, resultados, discussão e conclusão.

### 31 de agosto — início da tarde

- finalizar introdução, resumo e referências;
- montar no formato PROTEC;
- compilar e realizar revisão científica/visual.

### 31 de agosto — com margem antes do encerramento

- receber validação do orientador quando disponível;
- corrigir apenas problemas críticos;
- conferir 2 MB;
- submeter e guardar comprovante.

## 14. Divisão de responsabilidade

### O que pode ser executado pelo Codex

- preparar o runner e a estrutura de evidências;
- reexecutar os experimentos disponíveis no ambiente;
- identificar e corrigir inconsistências técnicas;
- gerar tabelas/figuras a partir dos dados reais;
- criar o novo projeto LaTeX;
- redigir e revisar as seções;
- compilar, comprimir e inspecionar o PDF.

### O que depende do usuário/orientador

- confirmar modalidade, título e dados pessoais no eCampus;
- autorizar a seleção final da narrativa quando houver conflito com o plano aprovado;
- validar se o caso CartPole representa adequadamente o trabalho realizado;
- conferir autoria, agradecimentos e informações institucionais;
- efetuar ou acompanhar a submissão no perfil autorizado.

## 15. Definição de pronto

O relatório estará pronto apenas quando:

- todas as caixas administrativas estiverem confirmadas;
- a matriz afirmação–evidência não tiver item central vermelho;
- as divergências Q8.8 e correção direcional estiverem explicadas;
- os bounds e o grafo da rede usados nas provas não excluírem estados possíveis;
- as métricas da Tanh e do treinamento forem calculadas a partir dos arquivos reais, sem valores fixos do webapp;
- os resultados forem reproduzíveis a partir do manifesto;
- nenhum gráfico experimental usar dados simulados;
- cada objetivo for respondido na conclusão;
- o PDF compilar, estiver visualmente correto e tiver até 2 MB;
- o arquivo submetido e o comprovante forem preservados.

Esse processo reproduz o traço mais forte observado nos premiados: uma cadeia curta entre problema relevante, método claro, evidência quantitativa, limitação honesta e conclusão diretamente sustentada.
