#include "mlp_weights.h"

// Correct ESBMC built-in intrinsics
extern float __VERIFIER_nondet_float();
extern void __VERIFIER_assume(int cond);
extern void __VERIFIER_assert(int cond) {
    if (!cond) {
        int *p = 0;
        *p = 0; // Fails verification
    }
}

int main() {
    // 1. Simular entradas analógicas dinâmicas
    float x1 = __VERIFIER_nondet_float();
    float x2 = __VERIFIER_nondet_float();
    
    // 2. Assumir que as features variam de 0.0 a 1.0 (Região válida do sensor)
    __VERIFIER_assume(x1 >= 0.0f && x1 <= 1.0f);
    __VERIFIER_assume(x2 >= 0.0f && x2 <= 1.0f);
    
    // 3. Vamos testar todos os Neurônios da Camada Oculta dinamicamente
    extern int __VERIFIER_nondet_int();
    int n = __VERIFIER_nondet_int();
    
    // Garantir que é um índice válido para o número de neurônios reais (0 a 3, total de 4)
    __VERIFIER_assume(n >= 0 && n <= 3);
    
    // y = (x1 * w1) + (x2 * w2) + bias
    float sinal_interno = (x1 * w_hidden[n][0]) + (x2 * w_hidden[n][1]) + b_hidden[n];
    
    // 4. A Hipótese: "Esse neurônio é inútil (MORTOS)"
    // Se a soma interna nunca passar de 0.0, a ReLU descarta e ele sempre sai Zero.
    // Portanto, asserimos que o sinal_interno sempre será <= 0.0
    __VERIFIER_assert(sinal_interno <= 0.0f);
    
    return 0;
}
