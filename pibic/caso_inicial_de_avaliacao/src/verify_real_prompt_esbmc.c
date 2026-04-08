#define TESTING
#include "runq.c"

#define N 8
#define D 8
#define GROUP_SIZE 2

#ifdef VERIFY_ESBMC
int8_t nondet_int8();
#endif

int main() {
    GS = GROUP_SIZE;

    // 1. SIMULADOR DE PROMPT REAL: "Olá"
    // Em IA, a palavra "Olá" vira um token, que é traduzido para um Embedding (Vetor de Ativação)
    // Abaixo definimos o vetor real de ativação da IA, tirando o Nondet do escopo do Usuário.
    int8_t x_q[N] = { 42, -15, 8, 110, -55, 30, 0, 77 }; 
    float x_s[N / GROUP_SIZE];

    int8_t w_q[D * N];
    float w_s[(D * N) / GROUP_SIZE];

#ifdef VERIFY_ESBMC
    // Escalas fixadas (sem limite point-float agressivo)
    for (int i = 0; i < N / GROUP_SIZE; i++) x_s[i] = 1.0f;
    for (int i = 0; i < (D * N) / GROUP_SIZE; i++) w_s[i] = 1.0f;

    // 2. Os PESOS da rede neural AINDA SÃO NONDET:
    // Nós perguntamos ao ESBMC: "Para este prompt ESPECÍFICO ('Olá'), existe - no multiverso de todas 
    // as IAs - ALGUM modelo de pesos que faria a matemática estourar na Memória?"
    for (int i = 0; i < D * N; i++) { w_q[i] = nondet_int8(); }
#endif

    QuantizedTensor tensor_x = { .q = x_q, .s = x_s };
    QuantizedTensor tensor_w = { .q = w_q, .s = w_s };
    float xout[D];

    // 3. Roda a Multiplicação no Prompt Fixo vs Pesos Nondeterministicos
    matmul(xout, &tensor_x, &tensor_w, N, D);

    return 0;
}
