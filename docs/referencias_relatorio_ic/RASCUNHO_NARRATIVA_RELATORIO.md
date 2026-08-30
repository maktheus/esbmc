# Rascunho de narrativa para o Relatório Final PIBITI

**Projeto:** Verificação Formal de Sistemas de Inteligência Artificial Generativa e Controladores Neurais utilizando ESBMC  
**Estudo de caso recomendado:** controlador DDPG para o problema Cart-Pole  
**Status:** texto-base para revisão após a rodada experimental final

> Este documento é um rascunho de redação, não a versão final para submissão. As marcações **[CONFIRMADO]**, **[VALIDAR]** e **[PLACEHOLDER]** devem ser revisadas antes da incorporação ao relatório oficial.

## 1. Introdução — proposta de texto

### 1.1. Contexto e problema

**[CONFIRMADO — redação proposta]**

Redes neurais têm sido incorporadas a sistemas que interagem com o mundo físico, incluindo veículos autônomos, robôs e controladores industriais. Nesses sistemas, uma falha de decisão pode produzir consequências diferentes de um erro meramente informacional: uma ação inadequada pode levar à perda de estabilidade, à violação de limites físicos ou à operação fora das condições previstas. Por esse motivo, o desempenho médio obtido em simulações ou testes aleatórios não é suficiente, por si só, para caracterizar a segurança de um controlador neural.

O problema do pêndulo invertido sobre um carro, conhecido como Cart-Pole, constitui um estudo de caso controlado para investigar essa questão. O sistema possui um estado formado pela posição e velocidade do carro e pelo ângulo e velocidade angular do pêndulo. O controlador deve escolher uma força horizontal a partir desse estado, mantendo o pêndulo dentro de uma região operacional segura. Embora o modelo seja pequeno, ele combina dinâmica física, decisão não linear e restrição de atuação, reunindo características relevantes de sistemas ciberfísicos.

Neste trabalho, o controlador analisado é um ator treinado com o algoritmo Deep Deterministic Policy Gradient (DDPG). A rede utilizada no estudo de caso possui arquitetura 4–24–24–1, com 48 neurônios nas duas camadas ocultas. **[CONFIRMADO pelos arquivos do projeto; informar no relatório final o arquivo de pesos e sua identificação/hash.]** A saída é convertida em uma força limitada ao intervalo de operação do modelo Cart-Pole.

### 1.2. Lacuna e abordagem

**[CONFIRMADO — redação proposta]**

A verificação formal oferece uma forma complementar de analisar esse tipo de controlador. Em vez de testar apenas uma coleção finita de trajetórias, o verificador procura uma prova ou um contraexemplo dentro do domínio e do limite temporal especificados no modelo. No caso de um resultado satisfatório, a propriedade é demonstrada para o sistema codificado e para as hipóteses assumidas. No caso de uma falha, o resultado pode fornecer uma atribuição concreta de entrada que viola a propriedade. Um timeout, por sua vez, não é uma prova nem uma refutação: indica que o procedimento não concluiu dentro do limite estabelecido.

O ESBMC foi escolhido por permitir a verificação de programas C/C++ por meio de verificação de modelos limitada e solucionadores SMT. Como a aritmética de ponto flutuante pode tornar a análise mais custosa, o projeto emprega uma representação em ponto fixo Q8.8, com fator de escala 256. Os pesos, entradas e operações do controlador são convertidos para a representação quantizada; os harnesses C submetidos ao ESBMC executam essa aritmética inteira. **[CONFIRMADO no código de geração e nos harnesses; registrar no relatório final a versão do ESBMC, o solver utilizado e a configuração efetivamente executada.]**

Essa escolha introduz uma questão metodológica: a propriedade verificada é uma propriedade do controlador quantizado, e não automaticamente do controlador original em Float32. Por isso, a comparação entre as duas representações é parte do próprio estudo. O projeto registra uma amostra de 10.000 estados para estimar a diferença entre as saídas Float32 e Q8.8. **[CONFIRMADO no arquivo `quantization_report.json`; os valores devem ser recalculados e preservados com o conjunto de estados, semente e script utilizados.]**

### 1.3. Pergunta de pesquisa

**[PLACEHOLDER — confirmar com orientador]**

> Em que condições e com quais limitações o ESBMC consegue provar, refutar ou diagnosticar propriedades de um controlador neural DDPG quantizado em Q8.8 para o sistema Cart-Pole, mantendo correspondência verificável entre o harness analisado e o simulador utilizado para reproduzir os resultados?

Essa pergunta é deliberadamente limitada. Ela não afirma que o ESBMC certifica o controlador em todos os estados, nem que a quantização preserva integralmente o comportamento em ponto flutuante. O foco é medir quais garantias são obtidas no domínio especificado, quais propriedades são refutadas por contraexemplos e quais permanecem inconclusivas por custo computacional.

### 1.4. Justificativa

**[CONFIRMADO como motivação; adaptar referências]**

A análise é relevante por três motivos. Primeiro, controladores neurais apresentam uma relação menos transparente entre estado, parâmetros e ação do que controladores clássicos, o que dificulta justificar seu comportamento somente por inspeção do código ou por testes amostrais. Segundo, uma prova ou um contraexemplo pode orientar a revisão do domínio operacional, da especificação de segurança ou da própria política de controle. Terceiro, a quantização é comum em implementações de baixo custo e sistemas embarcados, mas pode alterar a saída da rede; portanto, avaliar a diferença entre Float32 e Q8.8 evita atribuir ao controlador original uma propriedade que foi demonstrada apenas em uma implementação modificada.

O Cart-Pole não representa, por si só, um sistema industrial completo. Seu papel é oferecer um ambiente reproduzível no qual seja possível separar quatro elementos: a rede neural, a aritmética quantizada, a dinâmica física e as propriedades formais. Essa separação permite avaliar a cadeia de verificação antes de avançar para controladores ou modelos de maior porte.

### 1.5. Organização do relatório

**[PLACEHOLDER — utilizar após a estrutura LaTeX final]**

O restante do relatório está organizado da seguinte forma. A Seção 2 apresenta os objetivos e as definições das propriedades. A Seção 3 descreve o controlador, a quantização, a dinâmica do Cart-Pole, a geração dos harnesses e o protocolo experimental. A Seção 4 apresenta os resultados quantitativos e a classificação de cada propriedade como provada, refutada ou inconclusiva. A Seção 5 discute a relação entre os resultados formais e a simulação, as limitações do modelo e a comparação com trabalhos relacionados. A Seção 6 reúne as conclusões e os trabalhos futuros.

## 2. Objetivo geral — proposta de texto

**[PLACEHOLDER — conferir a redação oficial cadastrada no eCampus]**

Investigar a aplicabilidade do ESBMC à verificação de propriedades estruturais e de segurança de um controlador neural DDPG para o sistema Cart-Pole, utilizando aritmética de ponto fixo Q8.8 e um domínio operacional explicitamente definido, com avaliação quantitativa da diferença em relação ao controlador Float32 e reprodução independente dos resultados no simulador.

O objetivo usa “investigar a aplicabilidade” em vez de “provar a segurança do controlador” porque o estudo inclui propriedades que podem ser refutadas e outras que podem permanecer inconclusivas por timeout. Essa formulação é mais compatível com o significado da verificação de modelos limitada e com os resultados atualmente disponíveis.

## 3. Objetivos específicos — proposta de texto

**[CONFIRMADO/VALIDAR — a ordem pode ser ajustada ao plano de trabalho oficial]**

1. **Caracterizar o controlador neural.** Documentar a arquitetura 4–24–24–1, os pesos utilizados, as funções de ativação, o intervalo de força e os parâmetros do modelo DDPG.

2. **Construir a representação verificável.** Converter pesos e estados para Q8.8, explicitar a regra de arredondamento e truncamento e gerar harnesses C compatíveis com a semântica da aritmética do controlador executado.

3. **Quantificar a diferença de representação.** Comparar as saídas Float32 e Q8.8 em estados amostrados do domínio operacional, informando erro médio, percentis, máximo, intervalo de força, semente e procedimento de amostragem.

4. **Verificar propriedades estruturais da rede.** Investigar, para as duas camadas ocultas, se existem neurônios que nunca são ativados no domínio definido e se há neurônios permanentemente saturados, sem interpretar “ativação possível” como prova de utilidade ou qualidade do treinamento.

5. **Verificar o limite de atuação.** Testar formalmente se a força produzida pelo controlador quantizado permanece no intervalo admissível de -10 N a +10 N para todas as entradas admitidas pelo harness.

6. **Analisar segurança de um passo.** Verificar se a dinâmica discretizada e linearizada permanece dentro do limite angular após um passo, identificando e reproduzindo qualquer contraexemplo encontrado.

7. **Analisar a propriedade direcional.** Verificar, separadamente para inclinações positivas e negativas, se a ação produzida respeita a regra direcional definida na especificação. Classificar o resultado como provado, refutado ou inconclusivo, sem transformar timeout em sucesso.

8. **Reproduzir os resultados.** Injetar os contraexemplos no simulador e comparar o estado, a ação e a evolução resultante com os valores registrados na execução formal.

9. **Avaliar limites e escalabilidade.** Relacionar tempo de execução, tamanho da fórmula, solver, domínio, número de passos e granularidade da quantização, registrando quais propriedades não foram resolvidas dentro do limite experimental.

10. **Produzir uma cadeia auditável de evidências.** Associar cada afirmação central a um arquivo bruto, comando, versão, figura ou tabela, preservando os artefatos necessários para que outra pessoa possa repetir a análise.

## 4. Contribuições do trabalho — proposta de texto

### 4.1. Contribuições que podem ser afirmadas com base nos artefatos atuais

**[CONFIRMADO, sujeito à repetição final]**

Este trabalho apresenta as seguintes contribuições:

- uma cadeia de conversão de um ator DDPG para uma representação em ponto fixo Q8.8 e para harnesses C analisáveis pelo ESBMC;
- uma separação entre propriedades estruturais da rede e propriedades de malha fechada do controlador acoplado à dinâmica do Cart-Pole;
- uma avaliação quantitativa da diferença entre as saídas Float32 e Q8.8 em uma amostra de estados do domínio operacional;
- uma análise formal dos 48 neurônios das duas camadas ocultas, com registro individual dos estados “vivo”, “morto” ou “timeout”; **[VALIDAR se o relatório final usará exatamente os arquivos atuais ou uma nova execução]**;
- uma propriedade de limite de força que, no resultado atualmente registrado em `ddpg_closed_loop_results.json`, aparece como `SUCCESSFUL`; **[VALIDAR com nova execução e log bruto]**;
- contraexemplos formais para propriedades de direção e segurança em um passo, quando efetivamente confirmados pela mesma rodada experimental;
- um mecanismo de reprodução no simulador web, permitindo comparar o artefato verificado e a implementação que apresenta o comportamento ao usuário.

### 4.2. Formulações que devem permanecer como hipótese ou trabalho futuro

As seguintes afirmações não devem ser apresentadas como contribuições já demonstradas sem novos dados:

- “o controlador é seguro” em sentido global;
- “a quantização não introduz erro relevante” sem definir o limiar de relevância e tratar o erro máximo de aproximadamente 7,76 N registrado atualmente;
- “a cadeia possui fidelidade total” em relação ao modelo Float32;
- “o método escala para redes grandes”;
- “a verificação direcional foi concluída” enquanto existirem resultados conflitantes entre os arquivos JSON;
- “a ausência de neurônios mortos demonstra que o treinamento foi eficaz”.

A formulação tecnicamente defensável é que a mesma aritmética Q8.8 pode ser compartilhada pelo harness e pelo simulador, reduzindo a diferença entre esses dois artefatos quantizados. A diferença entre Q8.8 e Float32 continua sendo uma questão medida experimentalmente.

## 5. Resultados atuais que podem orientar a narrativa

Esta seção não substitui a seção de resultados do relatório final; ela registra o que a redação pode dizer provisoriamente e o que ainda precisa de confirmação.

### 5.1. Quantização

**[CONFIRMADO no `quantization_report.json`; repetir antes da submissão]**

O relatório de quantização atualmente armazenado registra fator de escala 256 e 10.000 estados amostrados. Os valores registrados são:

- erro absoluto médio: aproximadamente 0,0778 N;
- erro absoluto no percentil 95: aproximadamente 0,2262 N;
- erro absoluto no percentil 99: aproximadamente 1,0927 N;
- erro absoluto máximo: aproximadamente 7,7589 N;
- intervalo de força registrado: [-10, +10] N;
- erro relativo máximo registrado: aproximadamente 77,59%.

A leitura correta é assimétrica: a maioria dos estados da amostra apresenta erro menor, mas há estados extremos com discrepância elevada. O relatório deve incluir distribuição ou percentis, além do máximo, e deve explicar como os estados foram amostrados. Não se deve resumir esses valores apenas como “erro desprezível”.

### 5.2. Neurônios e saturação

**[CONFIRMADO no `ddpg_dead_neuron_results.json` e no `ddpg_saturation_results.json`; repetir ou anexar logs]**

Os arquivos atuais registram 24 neurônios vivos e nenhum morto na camada 1, e 24 neurônios vivos e nenhum morto na camada 2. Também registram nenhuma saturação permanente na camada 1 e uma saída classificada como responsiva.

O significado deve ser delimitado: para cada neurônio, “vivo” significa que foi encontrada pelo menos uma entrada admitida pelo domínio que o ativa. Isso não prova que o neurônio seja necessário, útil para o desempenho ou bem treinado. Analogamente, “não saturado” significa que a propriedade de saturação permanente não foi demonstrada naquele domínio; não é uma medida direta de qualidade da política.

### 5.3. Propriedades de malha fechada

**[VALIDAR — há divergência entre arquivos atuais]**

O arquivo `ddpg_closed_loop_results.json` registra `FAILED` para as duas propriedades direcionais, `FAILED` para a segurança em um passo e `SUCCESSFUL` para o limite de força. Outro arquivo, `closed_loop_results.json`, registra timeout para a propriedade direcional direita e um contraexemplo diferente para a esquerda. Os arquivos também não possuem, de forma suficiente, o comando completo, a versão do harness, a versão do solver e o log bruto que permitam escolher qual rodada é autoritativa.

Por essa razão, o texto final deve usar uma das seguintes formulações, conforme o resultado da rodada final:

> **Se a nova execução encontrar contraexemplo:** A propriedade direcional foi refutada no domínio especificado, por meio de um contraexemplo concreto. Isso significa que a implicação formalizada no harness não vale para todas as entradas admitidas; não significa que o controlador falha em todas as situações nem que o sistema é inutilizável.

> **Se a nova execução atingir timeout:** A propriedade direcional permaneceu inconclusiva dentro do limite de tempo adotado. O resultado não permite afirmar nem a satisfação nem a violação da propriedade.

> **Se a nova execução for satisfatória:** A propriedade foi provada apenas para o programa, a aritmética, o domínio e o bound codificados no harness. A conclusão não deve ser generalizada para o controlador Float32, para a dinâmica não linear completa ou para horizontes temporais não verificados.

Para a segurança em um passo, um contraexemplo deve ser descrito com estado inicial, ação, parâmetros da dinâmica, estado seguinte e critério de violação. A reprodução no simulador deve informar se a mesma quantização, truncamento e aproximação de ativação foram usados.

## 6. Discussão comparativa com trabalhos premiados

### 6.1. Base documental e limite da comparação

**[CONFIRMADO sobre disponibilidade das fontes]**

A comparação foi feita com as listas oficiais de premiação do CONIC/UFAM de 2021 a 2025. A profundidade é desigual: há texto integral diretamente relacionado para 2025, artigo integral derivado para 2021 e texto público com metodologia e resultados para 2024. Para 2022 e 2023, a análise está baseada principalmente no título oficial e em materiais posteriores ou relacionados. Portanto, não se deve afirmar que todos os vencedores possuem exatamente a mesma estrutura ou que um determinado fator causou a premiação.

### 6.2. Padrões observáveis

**[INFERÊNCIA fundamentada nos textos disponíveis]**

Os trabalhos melhor documentados compartilham uma sequência narrativa curta e verificável:

1. definem um problema técnico ou aplicado específico;
2. delimitam uma pergunta ou objetivo que pode ser respondido;
3. apresentam um método compatível com a pergunta;
4. comparam métodos, condições, modelos ou fatores;
5. mostram resultados quantitativos diretamente ligados aos objetivos;
6. interpretam as limitações e os resultados negativos;
7. encerram com uma conclusão que retorna ao problema inicial.

No estudo de vibrações de 2021, métodos analítico e numérico são comparados com curvas, retratos de fase e erro. No estudo de impressão 3D de 2024, o desenho Box-Behnken, a norma ASTM, a repetição experimental e a análise de fatores conectam entradas, medições e conclusões. No trabalho de visão computacional de 2025, a base própria, a comparação entre arquiteturas, o teste entre domínios e a discussão de generalização conectam um problema local a evidências quantitativas.

O ponto comum não é o tema ou a técnica específica. É a possibilidade de o avaliador seguir a cadeia “problema → método → evidência → consequência”. Essa interpretação é uma hipótese explicativa baseada nos documentos disponíveis, não uma declaração sobre os critérios internos dos jurados.

### 6.3. Como o projeto pode aplicar esse padrão sem imitação superficial

O projeto pode estabelecer uma cadeia equivalente:

| Elemento observado nos premiados | Aplicação ao Cart-Pole/ESBMC |
|---|---|
| Problema concreto | garantir ou diagnosticar propriedades de um controlador neural em um sistema físico simplificado |
| Pergunta delimitada | o que é provado, refutado ou não resolvido no domínio operacional definido? |
| Referência/comparação | Float32 versus Q8.8; harness versus simulador; propriedade versus contraexemplo |
| Artefato próprio | gerador de harnesses, pesos quantizados, especificações, logs e simulador |
| Evidência quantitativa | erro de quantização, número de neurônios, tempos, bounds, estados e contraexemplos |
| Limitação explícita | domínio finito, dinâmica linearizada, um passo, timeout e diferença Float32/Q8.8 |
| Impacto plausível | melhorar a rastreabilidade e o diagnóstico de controladores neurais em sistemas ciberfísicos |

Essa transposição deve ser feita no nível da lógica científica, e não pela cópia de frases ou da aparência dos trabalhos. O relatório não precisa alegar que resolveu a segurança geral do Cart-Pole; precisa mostrar com precisão quais garantias e diagnósticos o pipeline produziu.

## 7. Conclusão provisória — proposta de texto

**[VALIDAR após a nova execução experimental]**

Este estudo investigou uma cadeia de verificação formal para um controlador neural DDPG aplicado ao sistema Cart-Pole. A abordagem converteu o controlador para aritmética de ponto fixo Q8.8, gerou harnesses C para análise pelo ESBMC e utilizou uma implementação compatível no simulador para verificar os resultados observados. O uso de um estudo de caso pequeno permitiu separar propriedades da rede neural, propriedades da saída do controlador e propriedades da dinâmica de malha fechada.

Os artefatos atualmente disponíveis indicam que a análise estrutural encontrou entradas que ativam os 48 neurônios das duas camadas ocultas e não identificou saturação permanente na camada analisada. A comparação Float32–Q8.8, realizada em 10.000 estados registrados, indica erro médio baixo, mas também discrepância máxima elevada em estados extremos. Esse resultado reforça que a quantização deve ser tratada como parte da especificação do artefato verificado, e não como uma transformação neutra presumida.

Na malha fechada, o resultado atualmente registrado aponta prova do limite de força e contraexemplos para propriedades de segurança e direção. Entretanto, há divergência entre arquivos de resultados e os status finais só devem ser escritos depois de uma nova execução com logs, comandos, versões e limites de tempo preservados. A conclusão provisória, portanto, não deve declarar que o controlador é globalmente seguro. Ela deve afirmar que o ESBMC foi capaz de produzir pelo menos uma garantia sobre a saída e diagnósticos concretos para propriedades que não se sustentam — ou que não foram resolvidas dentro do limite computacional — no domínio codificado.

O principal resultado metodológico é a explicitação dos três estados possíveis da análise: propriedades provadas, propriedades refutadas por contraexemplos e propriedades inconclusivas por timeout. Essa distinção permite que a verificação formal funcione como instrumento de engenharia: uma prova documenta uma garantia sob hipóteses explícitas, um contraexemplo orienta a revisão do controlador ou do domínio e um timeout identifica uma limitação de escalabilidade. Em conjunto com a reprodução no simulador, essa classificação fornece uma base auditável para estudos posteriores com domínios particionados, maior número de passos, dinâmica não linear e outras arquiteturas de rede.

## 8. Checklist para converter este rascunho em texto final

- [ ] Confirmar modalidade, título, período, orientador e autores conforme o eCampus.
- [ ] Executar novamente todas as propriedades do Cart-Pole com uma única configuração autoritativa.
- [ ] Guardar versão do ESBMC, solver, sistema operacional, comando, timeout e timestamp.
- [ ] Resolver a divergência entre `ddpg_closed_loop_results.json` e `closed_loop_results.json`.
- [ ] Recalcular o relatório de quantização e preservar a amostra ou semente.
- [ ] Anexar logs brutos das verificações estruturais e de malha fechada.
- [ ] Diferenciar no texto “prova”, “contraexemplo”, “timeout” e “resultado desconhecido”.
- [ ] Não usar “fidelidade total” para descrever a relação Float32–Q8.8.
- [ ] Não afirmar estabilidade ou segurança global a partir de uma propriedade de um passo.
- [ ] Gerar figuras apenas a partir dos CSV/JSON finais.
- [ ] Vincular cada número a uma tabela, arquivo bruto ou comando reproduzível.
- [ ] Revisar a conclusão depois que os resultados conflitantes forem resolvidos.

