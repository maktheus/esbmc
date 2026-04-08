extern float __VERIFIER_nondet_float();
extern void __VERIFIER_assume(int cond);
extern void __VERIFIER_assert(int cond) {
    if (!cond) {
        int *p = 0;
        *p = 0;
    }
}

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

int main() {
    // 1. Em vez de puxar os pesos, geramos a matriz de pesos inteira de forma não-determinística
    float w_hidden[4][2];
    float b_hidden[4];
    float w_out[4];
    float b_out = __VERIFIER_nondet_float();
    
    // Assumimos que o hardware/framework inicializa pesos entre -2.0 e +2.0
    __VERIFIER_assume(b_out >= -2.0f && b_out <= 2.0f);

    for (int i = 0; i < 4; i++) {
        w_hidden[i][0] = __VERIFIER_nondet_float();
        w_hidden[i][1] = __VERIFIER_nondet_float();
        b_hidden[i] = __VERIFIER_nondet_float();
        w_out[i] = __VERIFIER_nondet_float();
        
        __VERIFIER_assume(w_hidden[i][0] >= -2.0f && w_hidden[i][0] <= 2.0f);
        __VERIFIER_assume(w_hidden[i][1] >= -2.0f && w_hidden[i][1] <= 2.0f);
        __VERIFIER_assume(b_hidden[i] >= -2.0f && b_hidden[i] <= 2.0f);
        __VERIFIER_assume(w_out[i] >= -2.0f && w_out[i] <= 2.0f);
    }
    
    // 2. Entradas normalizadas (0.0 até 1.0)
    float x1 = __VERIFIER_nondet_float();
    float x2 = __VERIFIER_nondet_float();
    __VERIFIER_assume(x1 >= 0.0f && x1 <= 1.0f);
    __VERIFIER_assume(x2 >= 0.0f && x2 <= 1.0f);

    // 3. Execução Arquitetural (Sem depender de um treino específico)
    float hidden_outputs[4];
    for (int i = 0; i < 4; i++) {
        hidden_outputs[i] = relu(x1 * w_hidden[i][0] + x2 * w_hidden[i][1] + b_hidden[i]);
    }
    
    float score = b_out;
    for (int i = 0; i < 4; i++) {
        score += hidden_outputs[i] * w_out[i];
    }
    
    // 4. Propriedade de Bounds Safety da ARQUITETURA
    // Com entradas [0,1] e todos os pesos e bias no envelope [-2, 2],
    // é impossível o neurônio final explodir para um score altíssimo.
    // O pior caso teórico manual = b_out(2) + 4*( 1*2 + 1*2 + 2 )*2 = 2 + 4*(6)*2 = 50.0.
    // Vamos asserir que a ARQUITETURA é fisicamente incapaz de passar de 55.0 na saída,
    // blindando o modelo contra Arithmetic Overflows sistêmicos, independente de estar treinado ou não!
    __VERIFIER_assert(score <= 55.0f && score >= -55.0f);

    return 0;
}
