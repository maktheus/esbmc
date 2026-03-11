# Verificação Formal de Redes Neurais Quantizadas (QNNs) com ESBMC-Python

Este relatório detalha o funcionamento de uma Rede Neural Quantizada (QNN), estabelece os resultados obtidos pelo framework QNNVerifier na literatura, e articula o plano de replicação e adaptação desse método para o ambiente Python nativo (`esbmc-python`), incluindo a simulação do motor do Frama-C (EVA).

---

## 1. O que é e omo funciona uma QNN?
Uma Rede Neural Artificial tradicional treina seus pesos sinápticos e viéses (*biases*) utilizando representação numérica de Ponto Flutuante (geralmente 32 ou 64 bits de ponto flutuante, previstos no IEEE 754). Isso entrega capacidade de simular distribuições muito suaves (curvas complexas), mas consome enorme potência de processamento e RAM.

Uma **QNN (*Quantized Neural Network*)** converte essa rede para **Ponto Fixo** com tamanho estrito de palavras de hardware (Ex: 8-bits ou 16-bits inteiros). 
O funcionamento consiste em:
1. **Escalonamento Geométrico:** O valor float real (ex: `0.5`) é multiplicado por um fator de escala (ex: `256` para 8-bits de fração) para se tornar num número Inteiro rígido (`128`).
2. **Aritmética Truncada:** Todas as somas e multiplicações entre os neurônios passam a utilizar registradores inteiros normais, o que é brutalmente mais rápido e econômico energeticamente na ponta (Edge AI / IoT).
3. **Realinhamento:** Sempre que dois números de ponto fixo são multiplicados (o que eleva a escala ao quadrado), os *bits* menos significativos são descartados (Shift-Right lógico `>>`) para alinhar a palavra de volta à escala de hardware.

**O Problema Verificado pelo SMT:** Esse truncamento geométrico perde a precisão infinitesimal da IA. O QNNVerifier se propõe a analisar, através do ESBMC, se a rede neural sofre degradação dessa quantização ao ponto de desclassificar uma predição crítica ou se sofrerá um `overflow` de memória sob ataque adversário.

---

## 2. O que foi feito e avaliado no Artigo (QNNVerifier)?
No paper *"QNNVerifier: A Tool for Verifying Neural Networks using SMT-Based Model Checking"*, os autores montaram uma pipeline:

1. Converteram modelos Keras/ONNX para código `C` sujo.
2. Construíram modelos operacionais (simuladores em macro C) para forçar o computador a interpretar cada soma/multiplicação neural como aritmética fixa pura (ex: `fxp_add`, `fxp_mult`).
3. Discretizaram as curvas complexas de ativação (como Sigmoid e Tanh) em velozes Tabelas de Busca Vetoriais (*Look-up Tables / LUTs*).

### Os Resultados Obtidos na Literatura (Para Replicarmos)
Os autores avaliaram 3 Datasets clássicos:
- **Iris:** Rede com 3 camadas em Tanh (classificação simples).
- **Character Recognition:** Rede com 4 camadas em Sigmoid.
- **AcasXu (Sistema anticolisão de aviões da FAA):** Rede grande de 6 camadas com 300 neurônios ReLU por camada.

**Metricas alcançadas que almejamos:**
- A correlação entre tempo de verificação (solução SMT) escala violentamente se a rede tiver mais que 16-bits. Quantização super agressiva (8-bits) provou acelerar as provas dos Solvers substancialmente na abordagem deles.
- A técnica mista (Frama-C + Look-Up Tables) permitiu que eles destronassem o estado-da-arte e trouxessem tempos mais rápidos do que a ferramenta otimizada de Oxford (Marabou) e empatassem com tecnologias aproximadas (Neurify).

---

## 3. Adaptação para Python (`esbmc-python`): A Simulação do Frama-C

Para replicar o estrondoso sucesso do `QNNVerifier` abandonando a linguagem C e indo para o Python puro via `esbmc-python`, nos deparamos com o gargalo do Frama-C.

O QNNVerifier depende do **Frama-C EVA (Evolved Value Analysis)** para domar o SMT Solver (ESBMC). Antes de entregar o arquivo C pro ESBMC, ele roda o Frama-C que lê todo o código, e infere os **intervalos matemáticos absolutos** (inferior e superior) que os valores da memória devem ter na vida real, e insere isso sob forma de `assume(variavel >= 0 e variavel <= 5)`. Isso poda os galhos inúteis da árvore de busca matemática, o que eles chamam de Injeção de Invariantes.

### Como Simularemos o Frama-C em Python?
O Python não possui um plugin mágico (EVA) equivalente. Faremos isso arquiteturalmente:
1. **Modelagem Abstrata no Transpilador:** O nosso próprio script conversor (`qnn_py_converter`) precisará aplicar **Interpretação Abstrata (Interval Arithmetic)** *offline*.
2. Quando lermos os pesos Keras, criaremos uma função que calcula o maior "estrago" que as matrizes numéricas atuais causariam em cada camada, desde a fronteira estrita da camada de entrada (`assume()`), propagando camada por camada.
3. Injetaremos explicitamente blocos literais `if not (X > min and X <= max): return` (os guardas semânticos substitutos do ESBMC_assume) direto no código fonte bruto em `.py` antes de enviá-lo ao `esbmc-python`.

### Pipeline ESBMC-Python do nosso Caso 1 Reescrito
O Caso 1 gerado em anexo (`mlp_qnn.py`) demonstra esse código-fim. 
1. Nele, a matemática Float foi destruída e substituída pela escala de *8-bits*. 
2. Simulamos o "Efeito Frama-C" cravando na rede `if not (h1 >= 0 and h1 <= 153)` porque `153` é a codificação em Ponto Fixo do pico máximo estrito `0.6` do neurônio do Caso 1, já conhecido.
3. Passamos a rede pro **Solver ESBMC**, que agora avalia apenas as multiplicações inteiras nos túneis matemáticos deixados pela injeção das bordas quantizadas.
