# Análise dos trabalhos premiados — Engenharias/Computação, CONIC UFAM 2021–2025

Análise realizada em 30 de agosto de 2026 para orientar a redação do Relatório Final PIBIC 2025/2026 (correção de identificação: o programa é PIBIC/PROPESP, não PIBITI/PROTEC — ver `PENDENCIAS_FINAIS_AUDITORIA.md`). O objetivo não é imitar temas ou textos, mas identificar padrões de comunicação científica e de evidência que possam ser aplicados honestamente ao projeto atual.

## Como esta análise foi feita

Foram usadas as listas oficiais de premiação do CONIC/UFAM de 2021 a 2025, armazenadas em `premiacoes_oficiais/`. A profundidade da análise varia conforme a disponibilidade pública do texto:

- **texto integral diretamente relacionado:** trabalho CONIC de 2025;
- **artigo integral derivado da mesma pesquisa:** vencedor de 2021;
- **texto público do artigo relacionado, com métodos e resultados:** vencedor de 2024;
- **título oficial e publicações posteriores/linha de continuidade:** vencedor de 2022;
- **título oficial e descrição do problema:** vencedor de 2023.

Consequentemente, as conclusões sobre 2021, 2024 e 2025 têm evidência mais forte. As observações sobre 2022 e 2023 são hipóteses prudentes e não afirmações sobre o conteúdo integral dos relatórios.

## Evidências por ano

### 2021 — vibrações com amortecimento não linear

**Trabalho premiado:** “Análise da dinâmica de vibrações com amortecimento não-linear por meio do método de Krylov-Bogoliubov”, de Robert Batista Neves, orientado por Gustavo Cunha da Silva Neto.

O artigo derivado disponível em `trabalhos_premiados/2021_Robert_Neves_artigo_derivado_COBEM2023.pdf` segue uma linha científica clara:

1. situa o problema e explica por que o amortecimento não linear exige tratamento específico;
2. declara um objetivo geral e objetivos específicos;
3. apresenta as hipóteses e o domínio de validade do modelo;
4. compara três abordagens: Runge–Kutta, Krylov–Bogoliubov e uma formulação refinada;
5. usa curvas temporais, retratos de fase e erro quadrático médio para responder ao objetivo;
6. mostra onde a aproximação funciona e onde perde qualidade, em vez de ocultar a limitação;
7. conclui retomando exatamente o desempenho relativo dos métodos.

**Provável fator de destaque:** triangulação. Um método numérico de referência permite avaliar as aproximações analíticas, e a conclusão é sustentada por gráficos e erro quantitativo. O texto deixa claro não apenas que o método funciona, mas sob quais condições.

Fontes públicas: [lista oficial de premiação](https://propesp.ufam.edu.br/images/conic/XXX_CONIC/RESULTADO_FINAL_PREMIADOS_XXX_CONIC.pdf) e [artigo derivado no COBEM 2023](https://www.abcm.org.br/proceedings/view/COB2023/1740).

### 2022 — compósitos com fibras amazônicas

**Trabalho premiado:** “Produção de compósitos poliméricos reforçados com fibras de juta e malva pós tratamento de hornificação”, de Raquel de Sousa Freire, orientada por Virginia Mansanares Giacon.

O texto integral do relatório premiado não foi localizado publicamente. O título e a continuidade da linha permitem observar, com cautela:

- problema material bem delimitado: efeito de um tratamento específico;
- variáveis e materiais comparáveis: juta, malva e tratamento de hornificação;
- conexão regional e de sustentabilidade pelo uso de fibras amazônicas;
- continuidade científica posterior, indício de que a investigação gerou uma linha aproveitável além da apresentação.

**Hipótese de destaque, não confirmação:** relevância regional somada a comparação experimental objetiva e continuidade da pesquisa. Não é possível atribuir métricas ou uma estrutura detalhada ao relatório sem o texto integral.

Fontes públicas: lista oficial armazenada nesta pasta e [publicação posterior relacionada](https://4spepublications.onlinelibrary.wiley.com/doi/abs/10.1002/pc.70462).

### 2023 — estimação de parâmetros em biotransferência de calor

**Trabalho premiado:** “Estimativa de parâmetros em problema de biotransferência de calor”, de Rafael Pereira Bezerra, orientado por Nilton Pereira da Silva.

O relatório integral não foi localizado publicamente. Pelo título, há três qualidades verificáveis na formulação do trabalho: um problema inverso específico, parâmetros como saída mensurável e um modelo físico explícito como base de avaliação.

**Hipótese de destaque, não confirmação:** formulação matemática nítida e possibilidade de validar parâmetros estimados contra referência, simulação ou dados. Não se deve afirmar qual algoritmo, conjunto de dados ou métrica foi usado sem o documento integral.

Fonte: lista oficial em `premiacoes_oficiais/CONIC_2023_XXXII_premiacao.pdf`.

### 2024 — parâmetros de impressão 3D por Box–Behnken

**Trabalho premiado:** “Estudo da influência dos parâmetros de impressão 3D nas propriedades mecânicas de material compósito polimérico reforçado com fibra de carbono através do projeto de experimentos Box-Behnken”, de Levi dos Santos Carneiro, orientado por Antonio do Nascimento Silva Alves.

O trabalho público relacionado apresenta:

- problema causal bem definido: quais parâmetros de impressão influenciam propriedades mecânicas;
- protocolo normalizado de ensaio, com referência à ASTM D638;
- projeto Box–Behnken com três fatores controlados: ângulo de deposição, preenchimento e temperatura do bico;
- 75 corpos de prova, 15 combinações e cinco réplicas;
- três variáveis de resposta: resistência à tração, resistência à ruptura e módulo de elasticidade;
- valores quantitativos e superfícies de resposta;
- conclusão hierarquizando os fatores: preenchimento foi o mais influente, seguido do ângulo; temperatura não apresentou influência significativa;
- discussão de dados anormais associados a falhas de fixação dos corpos de prova.

**Provável fator de destaque:** desenho experimental que conecta diretamente entrada, saída, repetição, norma e análise. O resultado não significativo da temperatura é tratado como achado, e anomalias são discutidas de forma transparente.

Fontes públicas: [anais do XXXIII CONIC](https://www.even3.com.br/anais/xxxiii-conic/1044490-estudo-da-influencia-dos-parametros-de-impressao-3d-nas-propriedades-mecanicas-de-material-composito-polimerico-/) e [texto relacionado do CONEM](https://www.researchgate.net/publication/383715622_Estudo_da_influencia_dos_parametros_de_impressao_3D_nas_propriedades_mecanicas_de_material_composito_polimerico_reforcado_com_fibra_de_carbono_atraves_do_projeto_de_experimentos_Box-Behnken).

### 2025 — visão computacional para riscos na rede elétrica da UFAM

**Trabalho premiado:** “Desenvolvimento de um sistema de detecção de riscos em redes elétricas da UFAM através de visão computacional com foco em árvores próximas aos condutores de distribuição de energia elétrica”, de Robson Marchegiani Seixas Nogueira, orientado por Luiz Eduardo Sales e Silva.

O texto integral em `trabalhos_premiados/2025_Robson_Nogueira_trabalho_CONIC.docx` tem quatro páginas e uma arquitetura muito econômica:

1. Resumo e palavras-chave;
2. Introdução;
3. Objetivo geral;
4. Metodologia;
5. Resultados e discussão;
6. Conclusões;
7. Referências e agradecimentos.

Os elementos mais fortes são:

- problema real e local, ligado à segurança da rede elétrica do campus;
- construção de base própria da UFAM e uso de base complementar do Google Maps;
- comparação de cinco arquiteturas/modelos, incluindo YOLOv8, InceptionV3 e MobileNetV2;
- teste cruzado entre domínios para avaliar generalização;
- tabelas com acurácias de treino/teste, em vez de adjetivos vagos;
- pipeline adicional de processamento de imagens para estimar distância;
- exposição de resultado negativo: um modelo analítico treinado na UFAM não generalizou bem para imagens do Google Maps, com justificativa ligada a iluminação e qualidade;
- conclusão que volta à utilidade local do sistema e ao valor da base produzida.

**Provável fator de destaque:** combinação de problema local importante, artefato demonstrável, base de dados própria, comparação quantitativa e honestidade sobre generalização. O texto não é longo e nem metodologicamente perfeito; sua força está na coerência entre problema, método, resultados e impacto.

Fontes públicas: [trabalho nos anais do XXXIV CONIC](https://doity.com.br/anais/xxxivconic2425/trabalho/505922) e lista oficial em `premiacoes_oficiais/CONIC_2025_XXXIV_premiacao.pdf`.

## O padrão comum entre os premiados

Não há acesso às fichas completas de avaliação nem base para afirmar uma fórmula causal de premiação. Ainda assim, os trabalhos mais bem documentados compartilham uma “espinha dorsal” observável:

| Padrão | Como aparece nos premiados | Aplicação ao projeto atual |
|---|---|---|
| Problema concreto e estreito | amortecimento não linear; parâmetros de impressão; árvores próximas a condutores | segurança de um controlador neural quantizado, sob um domínio operacional explícito |
| Pergunta que admite resposta | qual método aproxima melhor; qual parâmetro influencia; o sistema generaliza? | quais propriedades o ESBMC prova, refuta ou não resolve dentro do limite? |
| Referência ou comparação | Runge–Kutta; cinco classificadores; combinações Box–Behnken | Float32 × Q8.8, teste × prova formal, propriedade × resultado SMT |
| Evidência quantitativa | erro, acurácia, respostas mecânicas | erro em 10.000 estados, 48 neurônios, tempos, limites e contraexemplo |
| Artefato visível | curvas/modelos, corpos de prova, sistema de visão | harnesses C, pesos quantizados, relatórios JSON e simulador web |
| Limites reconhecidos | faixa de validade, má generalização, falha de fixação | timeout, hipótese de dinâmica linearizada, limite de um passo e quantização |
| Conclusão que responde ao objetivo | desempenho relativo e influência dos fatores | tabela final “provada/refutada/inconclusiva” e significado de cada resultado |
| Relevância além do experimento | engenharia, segurança, sustentabilidade, campus | confiabilidade de IA em sistemas ciberfísicos e diagnóstico reproduzível |

### O que provavelmente fez diferença

Com base nas evidências disponíveis, os fatores mais plausíveis são:

1. **coerência:** a conclusão responde à mesma pergunta apresentada na introdução;
2. **densidade de evidência:** cada afirmação importante tem tabela, figura, métrica, prova ou artefato correspondente;
3. **comparação justa:** existe referência, baseline, norma ou condição de controle;
4. **contribuição própria identificável:** base, método, sistema, experimento ou refinamento produzido pelo estudante;
5. **maturidade científica:** limitações e resultados negativos são interpretados, não escondidos;
6. **relevância fácil de explicar:** o leitor entende em poucas linhas por que o problema importa;
7. **economia narrativa:** poucos resultados fortes e conectados superam muitos casos frágeis ou dispersos.

## Diagnóstico comparativo do relatório atual

O projeto atual já possui ingredientes comparáveis aos trabalhos premiados, sobretudo no estudo CartPole:

- problema de segurança compreensível;
- controlador DDPG 4–24–24–1 e artefato executável;
- quantização Q8.8 compartilhada entre verificador e simulador;
- 48 neurônios analisados;
- limite de força formalmente provado;
- propriedade de segurança em um passo refutada por contraexemplo reproduzível;
- correção direcional com timeout declarado;
- amostra de 10.000 estados para comparação Float32 × Q8.8;
- aplicação web para demonstrar o resultado.

A desvantagem do relatório atual não é falta de conteúdo, mas dispersão e rastreabilidade insuficiente. Ele tenta sustentar muitos estudos de caso com níveis diferentes de maturidade e contém números sem arquivos brutos correspondentes. Isso enfraquece exatamente o padrão mais comum entre os premiados: uma cadeia curta e verificável entre pergunta, método, evidência e conclusão.

## Regra editorial resultante

Cada afirmação central do relatório final deve passar por quatro perguntas:

1. Qual objetivo ou pergunta ela responde?
2. Qual arquivo bruto, comando, tabela ou figura a comprova?
3. Qual é o domínio de validade da afirmação?
4. Qual limitação ou alternativa razoável precisa ser reconhecida?

Se uma frase não tiver resposta para as quatro, ela deve ser comprovada, enfraquecida ou removida. Essa disciplina aproxima o projeto do padrão científico observado nos premiados sem inventar resultados e sem transformar o relatório em propaganda.
