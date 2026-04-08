# Verificação Formal Avançada do Llama2: O Padrão "Compile-Time Memory Map"

Durante nossa jornada para validar as propriedades lógicas e limiares matemáticos (*Mathematical Thresholds*) da inferência Llama-2-7B usando o provador formal SMT (ESBMC), nos deparamos com o maior gargalo da Análise Estática de Sistemas Operacionais: **Hardware I/O**.

Model Checkers analíticos sofrem extrema punição de estado ao tentar simular bibliotecas dependentes do Kernel (como `mmap` ou alocações imensas de `fread` do disco). Quando injetamos o motor `run.c` puro do Llama para o ESBMC rodar, ele colapsa ao não conseguir mapear eficientemente os Gigabytes de pesos binários do repositório para o sistema linear formal.

### O Problema do `mmap` e Sistemas Lineares Formais (Entendendo o Gargalo)

Para entender essa limitação, precisamos lembrar como o **ESBMC** funciona por baixo dos panos. O ESBMC não "executa" o código linha por linha como um processador normal. Em vez disso, ele traduz o seu programa inteiro em uma **fórmula matemática gigante** (chamada de Sistema SMT - Satisfiability Modulo Theories). 

Quando o código C tradicional chega na linha:
`*data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);`

Aqui ocorrem dois grandes problemas para o Model Checker:
1. **Modelagem de Hardware e SO (Syscalls)**: Bibliotecas como o `mmap` interagem diretamente com o Kernel do Linux para gerenciar a paginação de hardware e o mapeamento de arquivos do Disco Rígido para a Memória Virtual. O ESBMC não tem um disco rígido embutido e nem gerencia paginação. Ele possui um modelo abstrato ("mock") muito simplório do Sistema Operacional. Quando ele encontra comandos obscuros do Kernel de alto nível ou I/O pesado de disco, ele não sabe como traduzir isso para álgebra linear.
2. **Explosão de Caminhos por Ponteiro Cego**: Como a simulação do `mmap` falha ou é limitada a arquivos minúsculos simulados, a array de pesos do modelo retorna como *Simbólica* (o ESBMC não consegue comprovar o valor real que estaria no disco). Consequentemente, ao executar as multiplicações de matriz em `forward()`, o ESBMC tenta explorar **todas as combinações possíveis** de números float32 para cobrir essa "incógnita" (Bilhões de caminhos falsos), causando uma falha catastrófica de memória no computador (State Explosion) ou travamento infinito.

## A Solução Arquitetural: "Memory-Baked Weights"

Para que nós finalmente não fiquemos reféns de arrays "Simulados e Falsos" (`nondet()`), foi necessário transcrever a Inteligência Artificial diretamente para a **Memória RAM** declarativa, em tempo de compilação.

### Passo 1: O Script Gerador e o Conversor (`bin_to_c.py`)
Criamos um conversor em Python que lê o arquivo `.bin` contendo a arquitetura Flutuante (Float32) real (mesmo que seja o `micro_model.bin` diminuto focado na rapidez do provador Z3) e gera **um Header nativo do C `model_data.h`**. 
Ele transcreve cada byte da rede neural como valores hexadecimais de um array estático (`const unsigned char model_data[] = {0x01, 0xFF..} `).

### Passo 2: O Bypass no Kernel C de Inferência (`verify_real_model_esbmc.c`)
Nossa Harness oficial importa o `model_data.h`. 
Nós construímos o struct `Config` extraindo os inteiros diretamente deste array interno. E o principal: o ponteiro `weights_ptr` do Llama é apontado para a memória nativa declarada.
**Como não há chamadas de Sistema Operacional**, o ESBMC herda instantaneamente o Modelo Real. O Solver assume que o programa *sempre teve esses dados armazenados estruturalmente*.

### Passo 3: Comprovação Matemática (A propriedade NP-Difícil)
Nós finalmente rodamos o `forward()` utilizando tokens constantes.
Com o LLM instanciado de forma 100% autêntica nativamente, alocamos nossa variável adversária **sem matar a rastreabilidade concreta**:

```c
// O modelo original roda nas memórias nativas reais
float output_orig = forward(&transformer, prompt_token, 0)[0];

// Injetamos um Ruído Simbólico no vetor Embedding fixo
int8_t noise = nondet_int8();
__ESBMC_assume(noise >= -1 && noise <= 1);
transformer.weights.token_embedding_table[...] += (float)noise;

// O ESBMC rastreia a propagação da incerteza até a ponta (Produto Final)
float output_adv = forward(&transformer, prompt_token, 0)[0];
__ESBMC_assert(abs(output_orig - output_adv) <= 2.0f, "Robustez Rompida");
```

## Execução
Todo o pipeline de geração de modelo, transcrição pra cabeçalho de memória e Limites de Desenrolamento estão consolidados.

```bash
# 1. Gerar Micro-Modelo em Binario (.bin)
python3 scripts/create_micro_model.py

# 2. Transcrever o .bin nativo para Memória C (.h)
python3 scripts/bin_to_c.py

# 3. Execução SMT Oficial via ESBMC 
./scripts/run_real_esbmc.sh
```

*(PS: A matemática em ponto flutuante via Z3 para matrizes de grandes dimensões pode gerar explosão de estado (*time-out*). O script `run_real_esbmc.sh` foi capado com Timeouts permissivos e Unwinds limitados aos Loops exatos das Multiplicações do Llama que a nossa Micro-Configuração gerou.)*
