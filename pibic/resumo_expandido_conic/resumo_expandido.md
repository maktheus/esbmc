---
title: "Verificação Formal de Sistemas de Inteligência Artificial Generativa e Controladores Neurais utilizando ESBMC"
author: "Matheus Serrão Uchôa (PIBITI 2025/2026) · Orientador: Prof. Dr. Iury Valente de Bessa"
date: "Universidade Federal do Amazonas · Faculdade de Tecnologia · Engenharia da Computação"
lang: pt-BR
---

**Resumo.** Controladores baseados em redes neurais podem apresentar bom desempenho em simulação e, ainda assim, não oferecer garantia verificável para todos os estados admitidos. Este trabalho investiga o uso do verificador ESBMC (*Efficient SMT-Based Context-Bounded Model Checker*) para analisar propriedades de um controlador neural DDPG aplicado ao sistema Cart-Pole, com arquitetura 4–24–24–1 e pesos representados em ponto fixo Q8.8. A comparação entre as versões Float32 e Q8.8 em 10.000 estados mostrou erro médio de 0,0778 N, mas erro máximo de 7,7589 N e inversão de sinal da ação no pior caso. O ESBMC com Z3 provou, em 3,5 s, que a força de saída permanece em |F| ≤ 10 N no harness analisado; a propriedade direcional permaneceu inconclusiva (*timeout*). O trabalho organiza um fluxo rastreável que distingue prova, refutação, reprodução de contraexemplo e resultado inconclusivo.

**Palavras-chave:** verificação formal; ESBMC; controlador neural; Cart-Pole; quantização Q8.8.

# 1. Introdução

Testes de simulação verificam apenas as execuções efetivamente selecionadas. Um controlador neural pode agir de forma adequada nas trajetórias testadas e produzir uma ação inadequada em regiões do espaço de estados nunca amostradas, problema agravado pela aritmética de precisão reduzida, em que arredondamentos e saturações alteram a fronteira entre ações. Métodos formais baseados em satisfatibilidade complementam os testes buscando, de forma sistemática e dentro de um limite, uma atribuição que viole uma propriedade (GOPINATH et al., 2021; MENG et al., 2022).

O objetivo não é declarar segurança irrestrita a partir de um único experimento, mas estabelecer quais garantias o verificador produz sob hipóteses explícitas, quais violações podem ser diagnosticadas por contraexemplos e quando o resultado permanece inconclusivo.

# 2. Objetivos

Avaliar, por meio do ESBMC, como a quantização Q8.8, a estrutura de um controlador DDPG e a verificação limitada de propriedades da planta Cart-Pole influenciam a produção de evidências formais reprodutíveis, e identificar os limites para reutilizar o fluxo em componentes de IA generativa. Os objetivos específicos são: (i) comparar Float32 e Q8.8 em um domínio definido; (ii) analisar a atividade dos neurônios; (iii) formalizar propriedades de segurança da malha fechada; (iv) registrar hipóteses, custos e *timeouts*; e (v) avaliar a generalização metodológica.

# 3. Metodologia

Estudo de caso experimental e reprodutível em quatro etapas: (i) identificação do artefato de controle; (ii) comparação numérica Float32 × Q8.8; (iii) especificação e verificação de propriedades no ESBMC com o solver Z3; e (iv) reprodução independente dos estados de contraexemplo no simulador. O ator DDPG (4–24–24–1, ReLU e tanh aproximada) foi convertido para Q8.8 (fator de escala 256) e traduzido em *harnesses* em C. A rodada canônica usou `RandomState(42)`, e comandos, ambiente, logs e hashes SHA-256 foram registrados em um manifesto. Um veredicto `SUCCESSFUL` é tratado como prova apenas dentro do domínio, do modelo e do horizonte declarados; `TIMEOUT` e `UNKNOWN` são resultados inconclusivos.

# 4. Resultados e Discussão

**Fidelidade da quantização.** Em 10.000 estados, o erro absoluto médio entre as saídas Float32 e Q8.8 foi de 0,0778 N; o percentil 95 foi 0,2262 N, o percentil 99 foi 1,0927 N e o máximo foi 7,7589 N. No pior estado, a ação inverteu de sinal (+6,4308 N em Float32 e −1,3281 N em Q8.8). A aproximação de tanh teve erro máximo de 0,0457 (4,57 % do intervalo unitário). Portanto, Q8.8 não deve ser tratado como equivalente a Float32 sem qualificar o domínio e a cauda de erro.

**Verificação formal.** O harness curto canônico para o limite de força |F| ≤ 10 N terminou `SUCCESSFUL` com Z3 em 3,5 s: é uma prova do programa quantizado e do domínio codificado, não do controlador Float32 nem da dinâmica não linear completa. O contraexemplo de segurança em um passo foi reproduzido como estado fixo, o que confirma a consistência do replay, mas não refuta sozinho a propriedade universal. A propriedade direcional ficou **inconclusiva** (*timeout* de 12 s com Z3; Boolector indisponível no executável utilizado).

**Estrutura neural.** Registros históricos indicam 24 neurônios vivos em cada camada oculta; entretanto, a auditoria identificou limites artificiais e independência indevida entre camadas, de modo que esse resultado é indicativo, não prova universal.

# 5. Conclusão

O ESBMC permite produzir evidências formais e rastreáveis sobre um controlador neural quantizado, mas apenas dentro de domínio, horizonte e hipóteses explícitos; os resultados não constituem certificação global. Como trabalho futuro: validar o checkpoint PyTorch, corrigir os *harnesses* estruturais, particionar o domínio, avaliar horizontes maiores e comparar a dinâmica linearizada com a não linear.

# Referências

GOPINATH, D.; MENG, K.; PASAREANU, C. S.; ZHANG, Y.; BESSA, I.; CORDEIRO, L. C. Verifying Quantized Neural Networks using SMT-Based Model Checking. In: IEEE/ACM International Conference on Automated Software Engineering (ASE), 36., 2021. IEEE, 2021.

MENG, K.; GOPINATH, D.; PASAREANU, C. S.; ZHANG, Y.; BESSA, I.; CORDEIRO, L. C. QNNVerifier: A Tool for Verifying Neural Networks using SMT-Based Model Checking. In: International Conference on Software Engineering (ICSE), 44., Companion Proceedings, 2022.
