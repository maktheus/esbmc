# Verificacao Formal de Controladores Neurais com ESBMC: Um Estudo de Caso com DDPG no Cart-Pole

---

## Resumo

Este trabalho apresenta uma metodologia de verificacao formal de controladores baseados em redes neurais artificiais utilizando o verificador de modelos ESBMC (Efficient SMT-Based Context-Bounded Model Checker). O estudo de caso aborda um controlador treinado pelo algoritmo Deep Deterministic Policy Gradient (DDPG) para o problema classico do pendulo invertido sobre carro (Cart-Pole), com acao continua no intervalo $[-10, +10]$ N. A rede neural do ator possui arquitetura 4-24-24-1 com ativacoes ReLU e Tanh. Para viabilizar a verificacao via solucionadores SMT de teoria de vetores de bits, empregamos quantizacao em ponto fixo Q8.8 (fator de escala 256) com divisao por truncamento no estilo C, garantindo fidelidade total entre a aritmetica verificada nos harnesses C e a aritmetica executada no simulador web. Definimos dois dominios de verificacao: (i) o dominio do modelo, que investiga propriedades estruturais da rede neural (neuronios mortos e saturacao), e (ii) o dominio do controlador, que investiga propriedades de malha fechada (correcao direcional, seguranca em um passo e limites de forca). Os resultados demonstram que todos os 48 neuronios estao ativos e nenhum se encontra saturado; a propriedade de limites de forca foi formalmente provada como satisfeita; a propriedade de seguranca em um passo foi refutada com um contraexemplo concreto reprodutivel; e a propriedade de correcao direcional resultou em timeout do solucionador. A aplicacao web desenvolvida em Next.js permite simulacao em tempo real, comparacao Float32 versus Q8.8 e injecao de contraexemplos, fechando o ciclo entre verificacao formal e validacao empirica. Este trabalho contribui para a area de verificacao de sistemas de inteligencia artificial aplicados a controle, demonstrando a viabilidade e os limites atuais de tecnicas de verificacao formal baseadas em BMC para controladores neurais em sistemas cibernetico-fisicos.

**Palavras-chave:** verificacao formal, redes neurais, ESBMC, verificacao de modelos limitada, aprendizado por reforco profundo, DDPG, Cart-Pole, ponto fixo, quantizacao, sistemas cibernetico-fisicos.

---

## 1. Introducao

A incorporacao de redes neurais profundas como controladores em sistemas cibernetico-fisicos (cyber-physical systems, CPS) representa um dos avancos mais significativos e, simultaneamente, um dos maiores desafios da engenharia de software moderna. Diferentemente de controladores classicos como PID, cujas propriedades de estabilidade e seguranca podem ser analisadas por metodos analiticos tradicionais (criterio de Nyquist, lugar das raizes, funcoes de Lyapunov), controladores neurais operam como funcoes nao-lineares de alta dimensionalidade cujo comportamento e extremamente dificil de ser caracterizado formalmente (Katz et al., 2017).

Em dominios criticos de seguranca --- veiculos autonomos, sistemas aeroespaciais, robotica cirurgica e controle industrial --- a mera validacao empirica por testes nao e suficiente para garantir a ausencia de falhas. A norma IEC 61508, o padrao DO-178C para software aeronautico e as diretrizes emergentes da Uniao Europeia para inteligencia artificial (EU AI Act) exigem niveis de asseguracao que so podem ser atingidos por metodos formais. A verificacao formal, entendida como a prova matematicamente rigorosa de que um sistema satisfaz uma especificacao, surge como abordagem complementar indispensavel.

O ESBMC (Gadelha et al., 2018; Cordeiro et al., 2012) e um verificador de modelos limitado (bounded model checker) baseado em Satisfiability Modulo Theories (SMT) que tem demonstrado desempenho competitivo na competicao internacional SV-COMP. O ESBMC converte programas C/C++ em formulas logicas e utiliza solucionadores SMT para determinar se existe alguma execucao que viola uma propriedade especificada. Quando tal execucao existe, o verificador retorna um contraexemplo concreto --- uma atribuicao de valores as variaveis de entrada que demonstra a violacao.

Este trabalho investiga a aplicacao do ESBMC a verificacao formal de um controlador neural treinado pelo algoritmo DDPG (Lillicrap et al., 2016) para o problema do Cart-Pole. As contribuicoes principais sao:

1. **Metodologia de verificacao em dois dominios**: separamos a verificacao do modelo neural (propriedades estruturais da rede) da verificacao do controlador (propriedades de malha fechada), seguindo principios de separacao de responsabilidades.

2. **Quantizacao Q8.8 com fidelidade total**: propomos um esquema de quantizacao em ponto fixo que elimina completamente a lacuna de fidelidade entre o artefato verificado e o artefato executado, incluindo uma aproximacao linear por partes da funcao tanh com 5 segmentos.

3. **Geracao automatica de harnesses**: desenvolvemos um pipeline Python que extrai pesos de modelos PyTorch, quantiza-os, propaga intervalos de pre-ativacao e gera harnesses C com todos os pesos expandidos inline, sem loops, para maximizar a eficiencia do solucionador SMT.

4. **Reproducao de contraexemplos**: os contraexemplos produzidos pelo ESBMC sao diretamente injetaveis no simulador web, onde reproduzem exatamente o comportamento previsto, validando a correcao da cadeia de verificacao.

O restante deste texto esta organizado como segue. A Secao 2 apresenta a fundamentacao teorica. A Secao 3 descreve a metodologia. A Secao 4 reporta os resultados. A Secao 5 discute as implicacoes. A Secao 6 conclui o trabalho.

---

## 2. Fundamentacao Teorica

### 2.1. Verificacao de Modelos Limitada e ESBMC

A verificacao de modelos limitada (Bounded Model Checking, BMC) foi proposta por Biere et al. (1999) como uma tecnica para verificar propriedades de seguranca em sistemas de transicao de estados. Dado um programa $P$, uma propriedade $\varphi$ e um limite de desdobramento $k$, o BMC constroi uma formula proposicional $\Phi_k$ tal que $\Phi_k$ e satisfativel se e somente se existe uma execucao de $P$ com ate $k$ passos que viola $\varphi$. Se a formula e satisfativel, o solucionador retorna uma atribuicao de variaveis que constitui o contraexemplo; se e insatisfativel, a propriedade vale para todas as execucoes de ate $k$ passos.

O ESBMC estende o BMC classico com suporte a teorias SMT, incluindo aritmetica de vetores de bits (bitvectors), aritmetica inteira, aritmetica de ponto flutuante (IEEE 754) e teoria de arrays. Para o presente trabalho, utilizamos a teoria de vetores de bits com o solucionador Boolector (Niemetz et al., 2018), reconhecido por sua eficiencia em aritmetica inteira. O ESBMC oferece tres primitivas centrais para a construcao de harnesses de verificacao:

- `nondet_int()`: declara uma variavel simbolica que representa todos os valores inteiros possiveis, formalizando o quantificador universal $\forall x \in \mathbb{Z}$;
- `__ESBMC_assume(cond)`: restringe o espaco de busca, descartando caminhos onde `cond` e falso, sem alterar a propriedade verificada;
- `__ESBMC_assert(cond, msg)`: especifica a propriedade a verificar; o ESBMC tenta encontrar uma atribuicao que satisfaz todos os assumes mas viola o assert.

A flag `--no-unwinding-assertions` desabilita as assertions automaticas de guarda de desdobramento de loops. Esta opcao e segura quando o harness nao contem loops, como e o caso de nossos harnesses lineares onde cada neuronio e expandido como uma linha individual de codigo C.

### 2.2. Aprendizado por Reforco Profundo e DDPG

O Deep Deterministic Policy Gradient (Lillicrap et al., 2016) e um algoritmo de aprendizado por reforco para espacos de acao continuos que combina ideias do DPG (Silver et al., 2014) com tecnicas de deep learning. O DDPG emprega uma arquitetura ator-critico com quatro redes neurais:

- **Ator** $\mu_\theta(s)$: mapeia estados em acoes continuas;
- **Critico** $Q_\phi(s, a)$: estima o valor Q da dupla estado-acao;
- **Ator-alvo** $\mu_{\theta'}$ e **Critico-alvo** $Q_{\phi'}$: copias suavizadas via Polyak averaging ($\tau = 0.005$) para estabilizar o treinamento.

A atualizacao do critico minimiza o erro quadratico medio:

$$\mathcal{L}_{\text{critic}} = \mathbb{E}\left[\left(Q_\phi(s, a) - \left(r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))\right)\right)^2\right]$$

A atualizacao do ator maximiza o valor Q esperado via gradiente de politica deterministica:

$$\nabla_\theta J \approx \mathbb{E}\left[\nabla_a Q_\phi(s, a)\big|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)\right]$$

Para exploracao, emprega-se ruido de Ornstein-Uhlenbeck, um processo estocastico correlacionado temporalmente que gera trajetorias de exploracao suaves, adequadas a controle de sistemas fisicos:

$$dx_t = \theta_{OU}(\mu - x_t) dt + \sigma \, dW_t$$

com $\theta_{OU} = 0.15$, $\mu = 0$ e $\sigma = 0.3$ em nossa implementacao.

### 2.3. O Problema do Cart-Pole

O Cart-Pole (Barto et al., 1983) consiste em um pendulo articulado no topo de um carro que se desloca sobre um trilho horizontal. O estado do sistema e descrito pelo vetor $\mathbf{s} = [x, \dot{x}, \theta, \dot{\theta}]^\top$, onde $x$ e a posicao do carro (m), $\dot{x}$ a velocidade linear (m/s), $\theta$ o angulo do pendulo com a vertical (rad) e $\dot{\theta}$ a velocidade angular (rad/s). A acao de controle e uma forca horizontal $F \in [-10, +10]$ N aplicada ao carro.

As equacoes de movimento nao-lineares, derivadas pela mecanica lagrangiana, sao:

$$\ddot{\theta} = \frac{g \sin\theta - \cos\theta \cdot \frac{F + m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p}}{l \left(\frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p}\right)}$$

$$\ddot{x} = \frac{F + m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p} - \frac{m_p l \ddot{\theta} \cos\theta}{m_c + m_p}$$

com os parametros: gravidade $g = 9.8$ m/s$^2$, massa do carro $m_c = 1.0$ kg, massa do pendulo $m_p = 0.1$ kg, semi-comprimento do pendulo $l = 0.5$ m e passo de integracao de Euler $\Delta t = 0.02$ s. A condicao de falha e $|x| > 2.4$ m ou $|\theta| > 12°$.

Para a verificacao formal com dinamica linearizada, aplicamos a aproximacao de pequenos angulos ($\sin\theta \approx \theta$, $\cos\theta \approx 1$), valida no dominio seguro $|\theta| \leq 12° \approx 0.21$ rad (erro relativo da linearizacao inferior a 0.7%).

### 2.4. Trabalhos Relacionados

A verificacao formal de redes neurais tem recebido atencao crescente na ultima decada. O trabalho seminal de Katz et al. (2017) introduziu o Reluplex, um procedimento de decisao SMT para redes com ativacoes ReLU, demonstrado no sistema ACAS Xu de previsao de colisao aeronautica. O Reluplex foi posteriormente aprimorado no Marabou (Katz et al., 2019), que incorporou otimizacoes como splitting on-demand e bound tightening.

Huang et al. (2017) propuseram tecnicas de verificacao de robustez baseadas em cobertura de camadas, enquanto Singh et al. (2019) desenvolveram o DeepPoly, um dominio abstrato poliedral para verificacao escalavel por interpretacao abstrata. O alpha-beta-CROWN (Wang et al., 2021) combinou branch-and-bound com propagacao linear para vencer a competicao VNN-COMP em multiplas edicoes.

No contexto especifico de verificacao por model checking, o QNNVerifier (Araujo et al., 2023) utilizou o ESBMC para verificar redes neurais quantizadas em ponto fixo, aplicando a metodologia ao benchmark ACAS Xu e demonstrando que a quantizacao permite evitar a teoria de ponto flutuante, significativamente mais custosa computacionalmente. Nosso trabalho estende essa linha ao aplicar a verificacao via ESBMC a um controlador neural em malha fechada com dinamica do sistema, introduzindo a verificacao de propriedades de seguranca que envolvem nao apenas a rede neural, mas tambem a evolucao temporal do sistema fisico.

Ivanov et al. (2019) propuseram verificacao composicional de sistemas neurais em malha fechada usando Verisig, que combina propagacao de alcancabilidade com abstraccoes polinomiais de funcoes de ativacao sigmoide. Sun et al. (2019) empregaram model checking para verificar redes recorrentes. Tran et al. (2020) desenvolveram o NNV, uma ferramenta de verificacao baseada em conjuntos estrela (star sets) para redes neurais feedforward e recorrentes em sistemas de controle.

Diferentemente desses trabalhos, nossa abordagem nao requer solucionadores especializados para redes neurais: utilizamos o ESBMC como verificador de modelos generico para programas C, onde a rede neural e os pesos sao expandidos inline como expressoes aritmeticas inteiras. Isso permite reusar toda a infraestrutura do ESBMC, incluindo suporte a multiplos solucionadores SMT e tecnicas de otimizacao como k-inducao.

---

## 3. Metodologia

### 3.1. Arquitetura do Controlador Neural

O controlador consiste na rede ator do DDPG com a seguinte arquitetura:

$$\mu_\theta: \mathbb{R}^4 \rightarrow [-10, +10] \subset \mathbb{R}$$

$$\mu_\theta(\mathbf{s}) = 10 \cdot \tanh\left(\mathbf{W}_3 \cdot \text{ReLU}\left(\mathbf{W}_2 \cdot \text{ReLU}\left(\mathbf{W}_1 \cdot \mathbf{s} + \mathbf{b}_1\right) + \mathbf{b}_2\right) + \mathbf{b}_3\right)$$

onde $\mathbf{W}_1 \in \mathbb{R}^{24 \times 4}$, $\mathbf{W}_2 \in \mathbb{R}^{24 \times 24}$, $\mathbf{W}_3 \in \mathbb{R}^{1 \times 24}$, $\mathbf{b}_1 \in \mathbb{R}^{24}$, $\mathbf{b}_2 \in \mathbb{R}^{24}$ e $\mathbf{b}_3 \in \mathbb{R}$. O total de parametros e $4 \times 24 + 24 + 24 \times 24 + 24 + 24 \times 1 + 1 = 769$. A funcao $\tanh$ na camada de saida, multiplicada por $F_{\max} = 10$ N, garante por construcao que a forca gerada esta contida no intervalo admissivel do atuador.

O treinamento foi realizado com taxa de aprendizado $10^{-3}$ para ator e critico, fator de desconto $\gamma = 0.99$, coeficiente de atualizacao suave $\tau = 0.005$, tamanho de batch 256 e buffer de replay com capacidade de $10^5$ transicoes.

### 3.2. Quantizacao Q8.8

A verificacao de redes neurais com pesos em ponto flutuante via solucionadores SMT com teoria de ponto flutuante IEEE 754 (flag `--floatbv` no ESBMC com solucionador Z3) e computacionalmente proibitiva para redes com centenas de parametros. A teoria de vetores de bits, por outro lado, e significativamente mais eficiente.

Para viabilizar a verificacao, empregamos quantizacao em ponto fixo Q8.8 com fator de escala $S = 256 = 2^8$. Cada valor real $v$ e convertido para inteiro via:

$$q(v) = \text{round}(v \times 256)$$

A aritmetica no dominio quantizado segue as regras:

- **Multiplicacao** de dois valores Q8.8 $a$ e $b$ que representam $a/S$ e $b/S$ respectivamente: o produto $a \times b$ representa $(a/S) \times (b/S) \times S^2$, necessitando divisao por $S$ para retornar ao formato Q8.8;
- **Divisao por truncamento** (C-style): $\text{cdiv}(a, b) = \text{trunc}(a/b)$, que trunca em direcao a zero, diferindo da divisao inteira de Python que arredonda em direcao a $-\infty$.

A correcao da aritmetica de truncamento e essencial para a soundness da verificacao. A funcao `c_div` em Python replica o comportamento da divisao inteira de C (`int(a/b)` em Python), e os harnesses C utilizam naturalmente a divisao nativa da linguagem. Qualquer discrepancia entre essas aritmeticas introduziria uma lacuna de fidelidade que invalidaria os resultados da verificacao.

### 3.3. Aproximacao Linear por Partes da Funcao Tanh

A funcao $\tanh$ e transcendental e nao pode ser representada exatamente em aritmetica de vetores de bits. Empregamos uma aproximacao linear por partes com 5 segmentos, projetada para minimizar o erro no dominio de operacao tipico do controlador:

$$\text{tanh}_{\text{pw}}(z) = \text{sgn}(z) \cdot f(|z|), \quad \text{onde}$$

| Segmento | Dominio $|z|$ (Q8.8) | Funcao (Q8.8)                     | Aproxima          |
|----------|----------------------|------------------------------------|--------------------|
| 1        | $[0, 64]$            | $\lfloor 252 |z| / 256 \rfloor$   | $\tanh \in [0, 0.245]$ |
| 2        | $(64, 192]$          | $62 + \lfloor 200(|z|-64)/256 \rfloor$ | $\tanh \in [0.245, 0.635]$ |
| 3        | $(192, 384]$         | $162 + \lfloor 92(|z|-192)/256 \rfloor$ | $\tanh \in [0.635, 0.905]$ |
| 4        | $(384, 768]$         | $231 + \lfloor 16(|z|-384)/256 \rfloor$ | $\tanh \in [0.905, 0.995]$ |
| 5        | $(768, \infty)$      | $255$                              | $\tanh \approx 1$   |

Os pontos de quebra foram escolhidos em $|z| = 0.25, 0.75, 1.5, 3.0$ (em escala real), correspondendo a regioes onde a curvatura da funcao $\tanh$ muda significativamente. O erro maximo da aproximacao e inferior a 3% em relacao a funcao $\tanh$ real.

Crucialmente, a **mesma** aproximacao linear por partes e utilizada tanto nos harnesses C verificados pelo ESBMC quanto no controlador TypeScript executado no navegador. Isso elimina a lacuna de fidelidade na funcao de ativacao de saida.

### 3.4. Analise de Erro de Quantizacao

A analise de erro foi conduzida sobre $n = 10{.}000$ estados amostrados uniformemente do dominio operacional. Comparando a saida em ponto flutuante Float32 ($F_{\text{float}}$) com a saida quantizada Q8.8 ($F_{\text{Q8.8}}$), obtivemos:

| Metrica                   | Valor       |
|---------------------------|-------------|
| Erro absoluto maximo      | 7.76 N      |
| Erro absoluto medio       | 0.08 N      |
| Erro absoluto percentil 95| 0.23 N      |
| Erro absoluto percentil 99| 1.09 N      |
| Erro relativo maximo      | 77.6%       |

O erro maximo ocorre em estados extremos do dominio (posicao e velocidade proximas aos limites), onde a combinacao de pesos grandes e truncamentos acumulados amplifica a discrepancia. Para o percentil 95, o erro e de apenas 0.23 N, indicando que a quantizacao preserva o comportamento do controlador na vasta maioria dos estados operacionais. E importante notar que a verificacao formal opera sobre o controlador quantizado --- nao sobre o controlador em ponto flutuante --- e, portanto, as propriedades verificadas referem-se ao artefato efetivamente executado.

### 3.5. Propagacao de Intervalos para Bounds de Pre-Ativacao

Para guiar o solucionador SMT e reduzir o espaco de busca, empregamos aritmetica de intervalos para computar sobreapromicacoes dos valores de pre-ativacao de cada neuronio. Para o neuronio $i$ da camada $l$ com pesos $w_{ij}$ e vies $b_i$:

$$\text{pre}_i = b_i + \sum_{j} w_{ij} \cdot h_j$$

Os bounds sao calculados analiticamente:

$$\text{pre}_i^{\min} = b_i + \sum_{j: w_{ij} \geq 0} w_{ij} \cdot h_j^{\min} + \sum_{j: w_{ij} < 0} w_{ij} \cdot h_j^{\max}$$

$$\text{pre}_i^{\max} = b_i + \sum_{j: w_{ij} \geq 0} w_{ij} \cdot h_j^{\max} + \sum_{j: w_{ij} < 0} w_{ij} \cdot h_j^{\min}$$

Apos a funcao ReLU: $h_i^{\min} = \max(0, \text{pre}_i^{\min})$, $h_i^{\max} = \max(0, \text{pre}_i^{\max})$. Esses bounds sao injetados nos harnesses como restricoes `__ESBMC_assume`, reduzindo drasticamente o espaco de busca do solucionador sem comprometer a soundness (as sobreapromicacoes contem todos os valores realmente alcancaveis).

A divisao por truncamento ao estilo C e utilizada na propagacao de intervalos para garantir consistencia com a aritmetica do harness. O uso de divisao por arredondamento inferior (floor), como e padrao em Python, produziria bounds mais apertados para termos negativos, potencialmente excluindo valores reais e comprometendo a soundness.

### 3.6. Geracao Automatica de Harnesses

O pipeline de geracao de harnesses e implementado em Python e opera da seguinte forma:

1. **Extracao**: os pesos sao extraidos do arquivo PyTorch (`.pth`) via `ddpg_weight_extractor.py`;
2. **Quantizacao**: cada peso e vies e convertido para Q8.8 via $q(v) = \text{round}(v \times 256)$;
3. **Propagacao de intervalos**: os bounds de pre-ativacao sao calculados para todas as camadas;
4. **Geracao de harness C**: o codigo C e gerado com todos os 769 pesos expandidos inline como constantes literais inteiras. Nao ha loops, arrays ou estruturas de dados --- cada neuronio e uma linha individual de codigo;
5. **Execucao do ESBMC**: o harness e submetido ao ESBMC com as flags `--no-unwinding-assertions --boolector` e timeout de 120 segundos;
6. **Parse do resultado**: o script Python analisa a saida do ESBMC, extrai contraexemplos e gera relatorios JSON.

A ausencia total de loops no harness e uma escolha deliberada. Loops em programas C sao desdobrados pelo ESBMC, o que introduz assertions automaticas de guarda cuja verificacao adiciona complexidade desnecessaria. Com 24 neuronios por camada e 3 camadas, o harness gerado contem aproximadamente 150 linhas de aritmetica inteira pura, constituindo um programa completamente linear que o solucionador SMT pode analisar sem o overhead do desdobramento de loops.

### 3.7. Dominios de Verificacao

Organizamos as propriedades verificadas em dois dominios distintos, seguindo o principio de separacao de responsabilidades:

**Dominio 1 --- Modelo (Estrutura da Rede Neural):**

- **Neuronios mortos**: para cada neuronio $i$, verifica-se se $h_i = \text{ReLU}(\text{pre}_i) = 0$ para todo estado valido. Se o ESBMC retorna `SUCCESSFUL`, o neuronio e provadamente morto (nunca ativa) e constitui candidato a poda. Se retorna `FAILED`, um contraexemplo demonstra a existencia de um estado que o ativa.

- **Saturacao**: para cada neuronio $i$, verifica-se se $\text{pre}_i > 0$ para todo estado valido. Se o ESBMC retorna `SUCCESSFUL`, o neuronio esta permanentemente saturado (a ReLU nunca corta). Para a saida, verifica-se se $z > 0$ sempre ou $z < 0$ sempre; se ambas falham, o controlador e responsivo (gera forcas em ambas as direcoes).

**Dominio 2 --- Controlador (Malha Fechada):**

- **Propriedade A (Correcao Direcional)**: $\theta > 0.10 \text{ rad} \wedge \dot{\theta} \geq 0 \Rightarrow F > 0$. Se o pendulo esta inclinado para a direita e sua velocidade angular nao o traz de volta, o controlador deve aplicar forca positiva (para a direita). Exploramos a monotonicidade da funcao $\tanh$: $z > 0 \Leftrightarrow \tanh(z) > 0 \Leftrightarrow F > 0$, o que permite verificar a propriedade sem necessidade de aproximar a $\tanh$.

- **Propriedade B (Seguranca em 1 Passo)**: para todo estado seguro $\mathbf{s}_0 \in S_{\text{safe}}$, apos 1 passo de integracao com dinamica linearizada ($\sin\theta \approx \theta$, $\cos\theta \approx 1$) e forca gerada pelo controlador quantizado, o angulo permanece seguro: $|\theta_1| \leq 12°$. A dinamica linearizada em Q8.8 e:

$$\ddot{\theta}_Q = \frac{4040 \cdot \theta_Q - 375 \cdot F_Q}{256}$$

$$\theta_Q' = \theta_Q + \frac{5 \cdot \dot{\theta}_Q}{256}, \quad \dot{\theta}_Q' = \dot{\theta}_Q + \frac{5 \cdot \ddot{\theta}_Q}{256}$$

onde os coeficientes 4040 e 375 codificam as constantes fisicas ($g$, $m_p$, $m_c$, $l$) em escala Q8.8, e 5 corresponde a $\Delta t = 0.02 \times 256 \approx 5$.

- **Propriedade C (Limites de Forca)**: $|F| \leq 10$ N para todo estado valido. Em Q8.8, isso equivale a $|F_Q| \leq 2560$. Esta propriedade decorre diretamente da saturacao da funcao $\tanh$ (cujo co-dominio e $(-1, 1)$) e do fator multiplicativo $F_{\max} = 10$.

### 3.8. Dominio de Entrada

O dominio de verificacao corresponde aos limites fisicos do sistema Cart-Pole, representados em Q8.8:

| Variavel       | Dominio real       | Dominio Q8.8          |
|----------------|--------------------|-----------------------|
| $x$            | $[-2.4, 2.4]$ m    | $[-614, 614]$         |
| $\dot{x}$      | $[-5.0, 5.0]$ m/s  | $[-1280, 1280]$       |
| $\theta$       | $[-12°, 12°]$ rad  | $[-53, 53]$           |
| $\dot{\theta}$ | $[-5.0, 5.0]$ rad/s| $[-1280, 1280]$       |

### 3.9. Aplicacao Web

A aplicacao web, desenvolvida em Next.js com TypeScript, serve como plataforma de validacao empirica e visualizacao. Seus componentes principais sao:

- **Simulacao em tempo real**: renderizacao Canvas do Cart-Pole com o controlador Q8.8 executando no navegador, utilizando a mesma aritmetica inteira e a mesma aproximacao de $\tanh$ dos harnesses verificados;
- **Comparacao Float32 vs Q8.8**: execucao paralela dos controladores em ponto flutuante e quantizado, com visualizacao das discrepancias;
- **Injecao de contraexemplos**: os estados produzidos pelo ESBMC como contraexemplos podem ser injetados diretamente como condicao inicial da simulacao, permitindo observar o comportamento previsto;
- **Graficos de estado**: 5 graficos em tempo real exibindo $x$, $\dot{x}$, $\theta$, $\dot{\theta}$ e $F$ ao longo do tempo;
- **Interface IController**: a arquitetura segue o principio SOLID com uma interface de controlador que permite intercambiar o tipo de controlador (DQN, DDPG, manual) sem alterar o restante do sistema.

---

## 4. Resultados

### 4.1. Dominio 1 --- Propriedades Estruturais do Modelo

**Neuronios Mortos:** Todos os 48 neuronios das camadas ocultas (24 na camada 1 e 24 na camada 2) foram verificados individualmente. Para cada neuronio $i$, o ESBMC tentou provar que $h_i = 0$ para todo estado do dominio. Em todos os 48 casos, a verificacao retornou `FAILED`, indicando que o ESBMC encontrou um contraexemplo --- uma atribuicao de entrada que ativa o neuronio. Portanto, **todos os 48 neuronios estao ativos** (alive). Nenhum neuronio morto foi detectado, indicando boa utilizacao da capacidade representacional da rede.

O tempo medio de verificacao por neuronio foi inferior a 5 segundos na camada 1 (expressoes com 4 termos) e inferior a 15 segundos na camada 2 (expressoes com 24 termos), utilizando o solucionador Boolector.

**Saturacao:** Todos os 24 neuronios da camada 1 foram verificados para saturacao permanente. Em todos os casos, o ESBMC retornou `FAILED`, indicando que nenhum neuronio esta permanentemente saturado --- todos possuem regioes de ativacao e desativacao no dominio de entrada. Para a camada de saida, verificou-se que $z > 0$ nem sempre e verdade e que $z < 0$ nem sempre e verdade, confirmando que a saida e **responsiva** --- o controlador efetivamente produz forcas positivas e negativas conforme o estado do sistema.

### 4.2. Dominio 2 --- Propriedades de Malha Fechada

Os resultados da verificacao de malha fechada sao sintetizados na tabela abaixo:

| Propriedade          | Descricao                              | Resultado     | Detalhes                      |
|----------------------|----------------------------------------|---------------|-------------------------------|
| A (Direcao, direita) | $\theta > 0.10, \dot\theta \geq 0 \Rightarrow F > 0$ | **TIMEOUT** | 120s insuficientes            |
| A (Direcao, esquerda)| $\theta < -0.10, \dot\theta \leq 0 \Rightarrow F < 0$ | **TIMEOUT** | 120s insuficientes            |
| B (Seguranca 1-step) | $\mathbf{s}_0 \in S_{\text{safe}} \Rightarrow |\theta_1| \leq 12°$ | **FAILED**  | Contraexemplo encontrado      |
| C (Limites de forca) | $|F| \leq 10$ N                        | **SUCCESSFUL**| Formalmente provado           |

**Propriedade A --- Correcao Direcional (TIMEOUT):** A verificacao nao convergiu dentro do limite de 120 segundos, tanto para a sub-propriedade "direita" quanto para a "esquerda". Isso indica que o espaco de busca, mesmo com a simplificacao oferecida pela monotonicidade de $\tanh$ (que elimina a necessidade de aproximar a funcao de ativacao), permanece demasiado extenso para o solucionador Boolector resolver neste timeout. O espaco de estados de entrada, mesmo restrito ao dominio de perigo ($\theta > 0.10$ rad), contem aproximadamente $614 \times 1280 \times (53-25) \times 1280 \approx 2.8 \times 10^{10}$ combinacoes discretas em Q8.8, antes da propagacao pelas 48 operacoes nao-lineares da rede.

**Propriedade B --- Seguranca em 1 Passo (FAILED):** O ESBMC encontrou um contraexemplo concreto:

$$x = -0.7539 \text{ m}, \quad \dot{x} = -3.9219 \text{ m/s}, \quad \theta = -0.1836 \text{ rad} \approx -10.5°, \quad \dot{\theta} = -1.5234 \text{ rad/s}$$

Este contraexemplo revela um cenario fisicamente plausivel: o pendulo esta proximo ao limite de falha ($-10.5°$, com limite em $-12°$) com velocidade angular negativa significativa ($-1.52$ rad/s). Mesmo com a forca maxima do controlador, a inercia do sistema e insuficiente para inverter o movimento em um unico passo de $\Delta t = 0.02$ s. A forca aplicada pelo controlador neste estado e fisicamente correta em direcao (forca negativa, pois $\theta < 0$), mas insuficiente em magnitude para superar a dinamica adversa. Este resultado nao indica uma falha do controlador, mas sim um limite fisico intrinseco: nenhum controlador com forca limitada a 10 N pode garantir seguranca em um unico passo a partir de estados extremos do dominio.

O contraexemplo foi injetado no simulador web e reproduziu exatamente o comportamento previsto --- o angulo excedeu $12°$ no passo seguinte --- validando a cadeia de verificacao.

**Propriedade C --- Limites de Forca (SUCCESSFUL):** O ESBMC provou formalmente que $|F_Q| \leq 2560$ (equivalente a $|F| \leq 10$ N) para todo estado do dominio de entrada. A prova decorre da composicao da saturacao da funcao $\tanh$ (aproximacao linear por partes limitada a $[-255, 255]$ em Q8.8, correspondendo a $[-1, 1]$ em escala real) com o fator multiplicativo $10 \times 256 / 256$. Este resultado fornece uma garantia formal de que o atuador nunca recebe comandos fora da faixa admissivel, independentemente do estado do sistema.

---

## 5. Discussao

### 5.1. Significancia dos Resultados

Os resultados obtidos demonstram tanto a viabilidade quanto as limitacoes atuais da verificacao formal de controladores neurais via BMC. A prova formal da Propriedade C (limites de forca) constitui uma garantia de seguranca certificavel: pode-se afirmar com certeza matematica que o controlador nunca comandara o atuador para alem de seus limites fisicos. Este tipo de garantia e diretamente relevante para certificacao em dominios como aeroespacial (DO-178C) e automotivo (ISO 26262), onde a saturacao de atuadores e uma condicao de seguranca fundamental.

A refutacao da Propriedade B, acompanhada de um contraexemplo concreto e reprodutivel, ilustra o poder diagnostico da verificacao formal. Enquanto testes aleatorios poderiam eventualmente encontrar falhas similares (exigindo milhares ou milhoes de simulacoes), a verificacao formal garante que, se existe uma violacao no dominio, ela sera encontrada --- ou que nao existe nenhuma. O contraexemplo identificado nao indica necessariamente uma deficiencia do controlador, mas revela as condicoes extremas sob as quais o sistema fisico, independentemente do controlador, nao pode manter a seguranca em um unico passo de tempo. Esse tipo de insight e valioso para engenheiros de sistemas, pois permite refinar os invariantes de operacao (por exemplo, restringir o dominio de estados iniciais permitidos).

### 5.2. Lacuna de Fidelidade Zero

Uma contribuicao metodologica central deste trabalho e a eliminacao da lacuna de fidelidade entre o artefato verificado e o artefato executado. Em muitos trabalhos de verificacao de redes neurais, a rede verificada formalmente opera em aritmetica de ponto flutuante idealizada, enquanto a rede executada em hardware pode sofrer erros de arredondamento, quantizacao e aproximacao de funcoes de ativacao. Neste trabalho, tanto os harnesses C verificados pelo ESBMC quanto o controlador TypeScript no navegador executam a mesma aritmetica inteira Q8.8, a mesma divisao por truncamento e a mesma aproximacao linear por partes de $\tanh$. A reproducao exata dos contraexemplos no simulador web confirma empiricamente que nao ha lacuna de fidelidade.

### 5.3. Escalabilidade e Timeouts

O timeout da Propriedade A revela um desafio fundamental da abordagem BMC para redes neurais: a complexidade da formula SMT gerada cresce combinatorialmente com o numero de neuronios e o tamanho do dominio de entrada. Mesmo com 48 neuronios ReLU (uma rede minuscula pelos padroes atuais de deep learning), a verificacao de propriedades que envolvem todo o dominio de entrada pode exceder o timeout pratico.

Estrategias para mitigar esse problema incluem: (i) decomposicao do dominio em sub-regioes menores e verificacao independente de cada uma (domain splitting); (ii) uso de solucionadores mais recentes como Bitwuzla, sucessor do Boolector; (iii) aumento do timeout; (iv) emprego de k-inducao para provar propriedades sem necessidade de enumerar todos os estados; e (v) preprocessamento do espaco de busca via tecnicas de bound tightening derivadas da comunidade de verificacao de redes neurais (Salman et al., 2019).

### 5.4. Verificacao Estrutural como Ferramenta de Diagnostico

A verificacao de neuronios mortos e saturacao, embora nao esteja diretamente relacionada a seguranca do sistema, oferece insight valioso sobre a qualidade do treinamento. A ausencia total de neuronios mortos e saturados indica que: (i) o treinamento foi eficaz em utilizar toda a capacidade representacional da rede; (ii) a inicializacao de pesos foi adequada; e (iii) a rede nao sofre do problema de dying ReLU (Lu et al., 2019). Essas verificacoes podem servir como criterio automatizado de qualidade apos o treinamento, complementando metricas empiricas como recompensa acumulada.

### 5.5. Comparacao com Abordagens Alternativas

Em comparacao com verificadores especializados como Marabou e alpha-beta-CROWN, a abordagem via ESBMC apresenta vantagens e desvantagens. Como vantagens, o ESBMC e um verificador generico que suporta C, C++, Python, Rust, CUDA e Solidity, permitindo verificar nao apenas a rede neural, mas tambem o codigo do sistema que a circunda (dinamica, interface de controle, logica de decisao). Alem disso, a quantizacao Q8.8 transforma o problema em aritmetica inteira pura, dominio onde os solucionadores SMT de vetores de bits sao extremamente eficientes.

Como desvantagem, a abordagem nao explora a estrutura especifica de redes neurais (linearidade por partes das ReLUs, simetria das camadas) como fazem ferramentas especializadas. Para redes com milhares de neuronios, como as utilizadas em sistemas de percepcao visual, a abordagem via BMC generico provavelmente nao escala. Entretanto, para controladores de baixa dimensionalidade como o presente, a abordagem demonstra-se viavel e praticamente util.

---

## 6. Conclusao

Este trabalho apresentou uma metodologia de verificacao formal de controladores neurais utilizando o verificador de modelos ESBMC, demonstrada em um estudo de caso com um controlador DDPG para o problema do Cart-Pole. As principais conclusoes sao:

1. **Viabilidade**: a verificacao formal de redes neurais de pequeno porte via BMC com quantizacao em ponto fixo e viavel e produz resultados em tempos praticos (segundos para propriedades estruturais, minutos para propriedades de malha fechada simples).

2. **Fidelidade**: a estrategia de quantizacao Q8.8 com mesma aritmetica no harness e no runtime elimina a lacuna de fidelidade, um problema critico em verificacao formal de sistemas baseados em aprendizado de maquina.

3. **Poder diagnostico**: o ESBMC produz contraexemplos concretos que podem ser diretamente reproduzidos em simulacao, fornecendo insight acionavel sobre os limites do controlador.

4. **Limites**: propriedades que envolvem dominos de entrada extensos e todo o grafo computacional da rede (como a correcao direcional) permanecem desafiadoras para o solucionador, resultando em timeout.

### 6.1. Trabalhos Futuros

Diversos caminhos de investigacao futura se abrem a partir deste trabalho:

- **Verificacao multi-passo**: estender a verificacao de 1 passo (Propriedade B) para $K$ passos de simulacao, utilizando desdobramento de loops ou k-inducao. O stub `closedloop_esbmc_stub.c` ja implementa a estrutura para $K = 50$ passos com dinamica linearizada, mas requer integracao com os pesos reais do controlador e estrategias de escalabilidade.

- **Aproximacoes nao-lineares da dinamica**: empregar polinomios de Taylor de ordem 3 ($\sin\theta \approx \theta - \theta^3/6$, $\cos\theta \approx 1 - \theta^2/2$) para capturar efeitos nao-lineares sem recorrer a funcoes transcendentais, aumentando a fidelidade da verificacao de malha fechada.

- **Decomposicao de dominio**: particionar o espaco de estados em sub-regioes e verificar cada uma independentemente, permitindo potencialmente resolver a Propriedade A dentro de timeouts praticos.

- **Redes maiores e mais profundas**: investigar a escalabilidade da abordagem para controladores com mais camadas e neuronios, incluindo arquiteturas convolucionais para controle baseado em visao.

- **Quantizacao adaptativa**: explorar esquemas de quantizacao com precisao variavel por camada (mixed-precision quantization), potencialmente reduzindo o erro de quantizacao sem aumentar significativamente a complexidade da verificacao.

- **Integracao com treinamento verificavel**: utilizar os resultados da verificacao formal como feedback durante o treinamento, penalizando politicas que violam propriedades de seguranca (verified training).

- **Comparacao sistematica**: conduzir uma comparacao experimental com ferramentas especializadas como Marabou, NNV e alpha-beta-CROWN no mesmo benchmark, avaliando trade-offs de generalidade, precisao e desempenho.

---

## Referencias

ARAUJO, R. G. et al. QNNVerifier: A Tool for Verifying Neural Networks using SMT-Based Model Checking. In: *Proceedings of the International Conference on Formal Methods in Computer-Aided Design (FMCAD)*, 2023.

BARTO, A. G.; SUTTON, R. S.; ANDERSON, C. W. Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Transactions on Systems, Man, and Cybernetics*, v. 13, n. 5, p. 834-846, 1983.

BIERE, A. et al. Symbolic Model Checking without BDDs. In: *Proceedings of the International Conference on Tools and Algorithms for the Construction and Analysis of Systems (TACAS)*, p. 193-207, 1999.

CORDEIRO, L. C.; FISCHER, B.; MARQUES-SILVA, J. SMT-Based Bounded Model Checking for Embedded ANSI-C Software. *IEEE Transactions on Software Engineering*, v. 38, n. 4, p. 957-974, 2012.

GADELHA, M. Y. R.; ISMAIL, H. I.; CORDEIRO, L. C. Handling Loops in Bounded Model Checking of C Programs via k-Induction. *International Journal on Software Tools for Technology Transfer*, v. 19, n. 2, p. 137-159, 2017.

GADELHA, M. Y. R. et al. ESBMC 5.0: An Industrial-Strength C Model Checker. In: *Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering (ASE)*, p. 888-891, 2018.

HUANG, X. et al. Safety Verification of Deep Neural Networks. In: *Proceedings of the International Conference on Computer Aided Verification (CAV)*, p. 3-29, 2017.

IVANOV, R. et al. Verisig: Verifying Safety Properties of Hybrid Systems with Neural Network Controllers. In: *Proceedings of the 22nd ACM International Conference on Hybrid Systems: Computation and Control (HSCC)*, p. 169-178, 2019.

KATZ, G. et al. Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks. In: *Proceedings of the International Conference on Computer Aided Verification (CAV)*, p. 97-117, 2017.

KATZ, G. et al. The Marabou Framework for Verification and Analysis of Deep Neural Networks. In: *Proceedings of the International Conference on Computer Aided Verification (CAV)*, p. 443-452, 2019.

LILLICRAP, T. P. et al. Continuous control with deep reinforcement learning. In: *Proceedings of the International Conference on Learning Representations (ICLR)*, 2016.

LU, L. et al. Dying ReLU and Initialization: Theory and Numerical Examples. *Communications in Computational Physics*, v. 28, n. 5, p. 1671-1706, 2020.

NIEMETZ, A.; PREINER, M.; BIERE, A. Boolector 3.0. In: *Proceedings of the International Conference on Computer Aided Verification (CAV)*, 2018.

SALMAN, H. et al. A Convex Relaxation Barrier for Neural Network Verification. In: *Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)*, 2019.

SILVER, D. et al. Deterministic Policy Gradient Algorithms. In: *Proceedings of the International Conference on Machine Learning (ICML)*, p. 387-395, 2014.

SINGH, G. et al. An Abstract Domain for Certifying Neural Networks. *Proceedings of the ACM on Programming Languages*, v. 3, n. POPL, p. 1-30, 2019.

SUN, X. et al. Formal Verification of Neural Network Controlled Autonomous Systems. In: *Proceedings of the 22nd ACM International Conference on Hybrid Systems: Computation and Control (HSCC)*, p. 147-156, 2019.

TRAN, H.-D. et al. NNV: The Neural Network Verification Tool for Deep Neural Networks and Learning-Enabled Cyber-Physical Systems. In: *Proceedings of the International Conference on Computer Aided Verification (CAV)*, p. 3-17, 2020.

WANG, S. et al. Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Neural Network Robustness Verification. In: *Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)*, 2021.
