# Entendendo o QNNVerifier

O **QNNVerifier** é uma ferramenta de software criada para garantir que redes neurais artificiais sejam seguras e confiáveis quando são colocadas para rodar no mundo real, especialmente em sistemas pequenos e com poucos recursos (como microcontroladores de IoT, drones, etc).

Este documento explica de forma simples e técnica o que foi atingido por essa ferramenta e como ela funciona, baseando-se no artigo *"QNNVerifier: A Tool for Verifying Neural Networks using SMT-Based Model Checking"*.

---

## 1. O Problema: Por que precisamos do QNNVerifier?

As redes neurais são treinadas em computadores super potentes usando números muito precisos (chamados de ponto flutuante, ou *floating-point*). No entanto, quando vamos colocar essa rede neural para funcionar em um dispositivo real e barato (como um sensor inteligente), precisamos "espremer" esse modelo. 

Esse processo de espremer o modelo se chama **quantização**. Nós trocamos a matemática precisa do *floating-point* por uma matemática de **ponto fixo** (*fixed-point*), que gasta menos bateria e memória. 

**O grande perigo:** Quando mudamos a matemática da rede neural, ela pode começar a errar! Além disso, redes neurais são vulneráveis a "ataques adversários" (pequenas mudanças invisíveis na entrada que fazem a rede errar drasticamente a resposta).

**O Objetivo (O que foi atingido):** O QNNVerifier foi criado para **provar matematicamente** que a versão espremida (quantizada) da rede neural em linguagem C é totalmente segura, não tem erros de *overflow* (estouro de memória) e é robusta contra ataques adversários. Ele é a primeira ferramenta de código aberto a conseguir verificar as redes neurais considerando os efeitos reais dessa quantização.

---

## 2. Como o QNNVerifier Funciona? (O Passo a Passo Técnico)

O funcionamento do QNNVerifier se assemelha a uma linha de montagem, que pega a rede neural pronta e a traduz para um problema matemático que o computador consegue resolver (decidir se é 100% seguro ou não).

Aqui está o detalhamento de como é feito, passo a passo:

### Passo 1: Tradução para Linguagem C
A ferramenta aceita redes neurais treinadas em diversos formatos populares (como Keras, TensorFlow, ONNX, NNET). O primeiro passo é converter essa rede neural inteira para código fonte na **linguagem C**. Em vez de rodar em um simulador ou em Python, a rede vira operações puras de soma e multiplicação no C. 

O script utilitário ensinado no README para isso é:
```bash
# Exemplo de conversão de NNET para C usando o utilitário nativo NNet2C
./NNet2C/generate.o <targeted_nnet_file> <target_generate_folder>
```

### Passo 2: Definição das Propriedades de Segurança
O usuário diz para o código C o que é considerado um comportamento "seguro". Por exemplo, as **propriedades de robustez**.
A ferramenta fornece a biblioteca `__ESBMC_assume` nativa e o `assert` padrão do C para o usuário configurar isso em código puro C.

Exemplo em código C (retirado do artigo) definindo uma área de atuação da rede:
```c
// Prepara entradas não-determinísticas (qualquer valor possível)
float x_1 = nondet_float();
float x_2 = nondet_float();

// Assuma o cenário base (Ex: restrição da entrada aceitável)
// A entrada x_1 deve ser entre 0 e 2. E a entrada x_2 deve ser entre -0.5 e 0.5.
assume(x_1 >= 0 && x_1 <= 2);
assume(x_2 >= -0.5 && x_2 < 0.5);

// Aqui existiria a execução da rede processando x_1 e x_2...

// Garanta matematicamente: O Output da classe 2 TEM que ser maior que a classe 1.
assert(y_2 > y_1); 
```

### Passo 3: Inferência de Intervalos (Ferramenta Frama-C)
Para facilitar a prova matemática, o QNNVerifier aciona um software chamado **Frama-C** (um analisador de código C). O Frama-C simula o código e descobre quais são os valores mínimos e máximos (intervalos / limites) que cada neurônio ou cada variável do programa pode atingir no pior cenário de execução. Para fazer isso automaticamente, o repositório usa:
```bash
# Executando o motor do Frama-c internamente (-f1 ou -f2) e exportando (-e1 ou -e2)
./Intervalgenerator.py -g
```

Ao fazer isso o QNNVerifier encontra a extremidade das variáveis por análise estática. Exemplo do resultado retornado pelo Frama-C que é injetado posteriormente:
```c
float x0 = Frama_C_float_interval(0, 60760);
float x1 = Frama_C_float_interval(-3.141592, 3.141592);
```
Ou no meio do código dos neurônios (reduzindo absurdamente o espaço matemático para o avaliador da frente analisar):
```c
// O modelo sabe de antemão que o neurônio zero da camada 1 jamais vai ser negativo 
// ou bater valores gigantes, enjaulando a matemática do problema.
__ESBMC_assume ((layer1[0] >= -0.0) && (layer1[0] <= 0.256393373013));
```

### Passo 4: Troca para Matemática Discreta e Ponto Fixo
A matemática de redes neurais é baseada em curvas (Funções de Ativação Não-lineares, como Tanh e Sigmoid). Computadores odeiam curvas contínuas para fazer provas lógicas. 
- O QNNVerifier troca essas curvas por **Tabelas de Busca (Look-up Tables)**, discretizando as funções com um erro máximo que não afeta o resultado, tornando a verificação absurdamente mais rápida.
- Também substitui as operações padrão originais em Ponto Flutuante pelas rotinas aritméticas seguras de **Ponto Fixo (Fixed-Point)** usando structs e macros do C para travar *overflow*. 

Script que converte o código flutuante do Passo anterior em ponto fixo seguro e truncado antes de rodar a verificação ESBMC:
```bash
./fxpgenerator_acas.py
```

Eis um exemplo de como fica o processamento C no baixo nível (tudo encapsulado usando diretivas puras de `fxp_t`):
```c
// Loop rodando as multiplicações de pesos de um Neurônio mas operando 100% em Ponto Fixo quantizado:
for (unsigned int i = 0; i < w_len; ++i) {
    fxp_t w_fxp = fxp_float_to_fxp(w[i]); // Converte float da matriz pra fxp_t (ponto fixo)
    fxp_t x_fxp = fxp_float_to_fxp(x[i]);
    // Soma(Resultado Anterior, Multiplicacao Fixa(Peso_fxp, Entrada_fxp))
    result = fxp_add(result, fxp_mult(w_fxp, x_fxp));
}
```

### Passo 5: Verificação de Modelos SMT (Ferramenta ESBMC)
O arquivo C, agora super otimizado e anotado, é entregue ao **ESBMC** (desenvolvido pela UFAM e Manchester). O ESBMC é um "Model Checker SMT (Satisfiability Modulo Theories)".
- Ele transforma absolutamente todas as linhas de código C e os *asserts* em uma única e gigantesca equação lógica.
- Ele joga essa equação em um solucionador matemático (SMT Solver, como Yices, Bitwuzla ou Boolector).
- O solucionador tenta de todas as formas quebrar o `assert()`. Se for possível quebrar (achar uma entrada que mude a classificação original), ele reporta **FALHAL** e mostra o contraexemplo (mostra qual pixel exato mudar para enganar a IA).
- Se for matematicamente impossível, ele reporta **SUCESSO**, garantindo que a rede neural quantizada é 100% à prova de falhas para aquele cenário de segurança!

---

## 3. Resumo das Inovações

O QNNVerifier se destaca e é importante por alguns motivos principais destacados no artigo:
1. **Verificação Direto no Código:** Diferente de outras ferramentas que verificam o modelo abstrato matemático, ele verifica o **código C real** que vai rodar no dispositivo da ponta.
2. **Suporte a Quantização:** É a primeira a conseguir verificar os problemas e as nuances da aritmética de ponto fixo.
3. **Velocidade:** Combinando o Frama-C (análise de intervalos) e discretização de funções curvas, ele acelera (em ordens de grandeza) o processo super pesado da prova de teoremas em IA, batendo o tempo de outras ferramentas famosas da literatura (como Neurify e Marabou).
