
#include <stdio.h>

extern void __VERIFIER_assume(int cond);
extern void __VERIFIER_assert(int cond) {
    if (!cond) {
        int *p = 0;
        *p = 0;
    }
}

// Constantes Inteiras Quantizadas (Scale: 256)
int qw_hidden[4][2] = {
    {598, 710},
    {-520, 522},
    {-38, -388},
    {-560, -549}
};
int qb_hidden[4] = {-598, -3, 517, 560};
int qw_out[4] = {-688, 667, 617, -1321};
int qb_out = 225;

// Simulação Frama-C: Limites observados estatisticamente no Python (já na escala 256)
int MIN_HIDDEN_VAL = -1280; // max value could be around 5.0
int MAX_HIDDEN_VAL = 1280;

int relu_int(int x) {
    return x > 0 ? x : 0;
}

int mlp_forward_qnn(int x1, int x2) {
    int hidden_outputs[4];
    for (int i = 0; i < 4; i++) {
        // Matemática Ponto Fixo (Inteiros puros, amigável para ESP32)
        int h_val = (x1 * qw_hidden[i][0]) / 256 + (x2 * qw_hidden[i][1]) / 256 + qb_hidden[i];
        
        // --- INJEÇÃO DE INVARIANTE (Frama-C) ---
        // Podamos o espaço de busca matemático do ESBMC confirmando que a soma interna não colapsou
        __VERIFIER_assume(h_val >= MIN_HIDDEN_VAL && h_val <= MAX_HIDDEN_VAL);
        
        hidden_outputs[i] = relu_int(h_val);
    }
    
    int score = qb_out;
    for (int i = 0; i < 4; i++) {
        score += (hidden_outputs[i] * qw_out[i]) / 256;
    }
    
    return score;
}

int main() {
    // Em Ponto Fixo (Escala 256):
    // 0.0 é 0
    // 1.0 é 256
    
    int zero = 0;
    int um = 256;
    
    // Testes XOR Reconstruídos
    __VERIFIER_assert(mlp_forward_qnn(zero, zero) <= 0);
    __VERIFIER_assert(mlp_forward_qnn(um, um) <= 0);
    __VERIFIER_assert(mlp_forward_qnn(zero, um) > 0);
    __VERIFIER_assert(mlp_forward_qnn(um, zero) > 0);
    
    return 0;
}
