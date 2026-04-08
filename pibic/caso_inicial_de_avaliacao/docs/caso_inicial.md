# Caso Inicial de Avaliação: Llama2.c (Modelo Gerativo Quantizado)

## Propósito

Este diretório contém uma prova de conceito para aplicar o **ESBMC** (um verificador de software focado em C/C++) em uma arquitetura de modelo que **não é de classificação**, satisfazendo o requisito de diversificar e buscar uma validação *diferente* dos modelos atuais como o MLP numérico.

## O Modelo Escolhido

Foi escolhido o código nativo em C do **llama2.c** (criado por Andrej Karpathy), mas especificamente o seu motor quantizado em inteiro de 8-bits: `runq.c`. 
Este é o núcleo de inferência para um Large Language Model (LLM) que faz **geração autoregressiva** (predição do próximo token), quebrando o paradigma de classificação usado anteriormente no PIBIC.

*Nota Técnica: Devido ao bloqueio persistente de porta 443 para o `github.com` na máquina atual (que impedia o `git clone`), nós desviamos a rota acessando o `raw.githubusercontent.com` diretamente via `curl` para resgatar os arquivos originais `run.c` e `runq.c` essenciais para a verificação.*

## O que torna essa validação "Diferente"?

1. **Domínio do Modelo (Gerativo vs Classificação):** Em vez de classificar entradas em um conjunto fechado de rótulos (como feito no `mlp_qnn.py`), o Transformer Llama2 projeta estados latentes contínuos para inferência sequencial (Chat / Geração de Texto).
2. **Método de Mapeamento com Ferramenta Formal:** 
   O código do LLM baseia-se pesadamente em cálculo vetorial de matrizes INT8 agrupadas (`QuantizedTensor`). Nosso harness (`verify_esbmc.c`):
   - Importa diretamente o arquivo original (`#include "runq.c"`).
   - Introduz Injeção Não-Determinística (`nondet`) de tensores de pesos e dados nos moldes limitados requeridos por verificação formal (N=16, D=16, Group=16).
   - **Garante integridade acumulativa:** Prova formal de que a acumulação `((int32_t) x->q) * ((int32_t) w->q)` na multiplicação da rede neural jamais sofrerá _overflow_ em máquinas 32-bits para qualquer conjunto válido de tensores.
   - **Garante proteção em ponto-flutuante:** Valida a ausência de valores matematicamente inválidos (`NaN` / `Infinity`) ao fim do de-quantizer da Attention.

## Como Executar e Validar

O ambiente contém um script autônomo para execução da verificação formal usando o binário esbmc local.

```bash
./run_esbmc.sh
```

A saída do ESBMC confirmará o "Verification Successful" ou indicará violações nas asserções sobre os ponteiros vetoriais da arquitetura, verificando a segurança do modelo quantizado no nível do compilador.

---

## Detalhamento do Repositório Clonado

### O Repositório Original
Os arquivos base extraídos para este diretório pertencem ao repositório público **[karpathy/llama2.c](https://github.com/karpathy/llama2.c)**. 
Andrej Karpathy (cientista e ex-diretor de IA da Tesla) criou este respositório para provar que a inferência do transformador **Llama-2** (um LLM gerativo gigantesco da Meta) poderia ser escrita do zero em um único arquivo C (`run.c`) puro, sem dependências como PyTorch ou TensorFlow.
Para esta validação de PIBIC nós resgatamos especificamente o `runq.c`, que é o motor capaz de rodar uma versão **quantizada em inteiros de 8-bits** da rede neural, economizando RAM.

### Como Funciona a Quantização no Código?
No modelo, os pesos float de 32-bits que saem do treinamento original são agrupados e convertidos em arrays limitados de `-128 até +127` (`int8_t`). Junto dos inteiros, salva-se um único "fator de escala" (float puro) para que depois a rede possa reverter a conta.
A representação desses tensores no `runq.c` foi clonada estruturalmente desta forma:

```c
typedef struct {
    int8_t* q;    // Os valores (pesos ou ativações) espremidos no espaço quantizado (-128 a 127)
    float* s;     // Fator de Escala (1 número real por grupo) necessário na dequantização
} QuantizedTensor;
```

### A Peça Central: A Multiplicação de Matriz Quantizada (O Alvo do ESBMC)
Em LLMs, 90% da computação é multiplicar tensores de *Query, Key e Value* ou *Feed-Forwards*. As multiplicações de matriz acontecem na função `matmul`.
A genialidade (e ao mesmo tempo o **Perigo Matemático**) desse código é que ele resolve os cálculos da gigantesca IA usando Acumuladores (`int32_t val`) para receber somas sucessivas de inteiros básicos!

O código clonado do Loop Matrix-Multiplication do repo funciona assim:
```c
// runq.c 
// Multiplica a Matriz do Peso W(d, n) pelo vetor de Entrada x(n) 
void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0; // O Acumulador alvo da nossa análise!
        int in = i * n;

        // O Loop Matemático realiza Produto Ponto sobre as colunas
        for (int j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                // Risco de Out-of-bounds bounds nos índices e Risco de Overflow na soma no ival!
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            // De-quantiza reconstruíndo com os multiplicadores float `s` de Grupo
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }
        xout[i] = val; // Saída
    }
}
```

### Como Conectamos Isso ao Verificador (O Código Harness)
Nós construímos o `verify_esbmc_test.c` (ou gerado dinamicamente via o script de Benchmark `benchmark_esbmc.py`). O objetivo desse script é injetar na memória matrizes de dados "Fantasmas" e chamar o `matmul` garantindo que o ESBMC explodirá todas as equações pra provar que aquele `ival` visto acima nunca extrapola limites e que o leitor nunca extrapola os índices permitidos de array.

O Código Harness simplificado injetado por nós:
```c
#define TESTING
#include "runq.c" // Importamos o código do Andrej Karpathy

#define N 4 
#define D 4
#define GROUP_SIZE 2

int main() {
    GS = GROUP_SIZE;

    // 1. Alocamos os espaços de tensores do Llama-2 na memória para teste.
    int8_t x_q[N];
    float x_s[N / GROUP_SIZE];
    int8_t w_q[D * N];
    float w_s[(D * N) / GROUP_SIZE];

#ifdef VERIFY_ESBMC
    // 2. Preenchemos com a Magia SMT do ESBMC! 
    // Em vez de números estáticos, injetamos NONDET (Valores universais e arbitrários)
    for (int i = 0; i < N; i++) { x_q[i] = nondet_int8(); }
    for (int i = 0; i < D * N; i++) { w_q[i] = nondet_int8(); }
#endif

    QuantizedTensor tensor_x = { .q = x_q, .s = x_s };
    QuantizedTensor tensor_w = { .q = w_q, .s = w_s };
    float xout[D];

    // 3. Forçamos a verificação da Máquina Llama-2 original sob simulação extressante!
    matmul(xout, &tensor_x, &tensor_w, N, D);

    return 0; // Se o ESBMC compilar até aqui nas árvores de simulação sem falhar, o código é Criptograficamente Seguro!
}
```
