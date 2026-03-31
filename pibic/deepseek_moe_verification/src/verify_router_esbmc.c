#include <stdio.h>
#include <math.h>

#include "../include/router_weights.h" 

// Declaracoes do ESBMC com Bool para funcionar no ESBMC 8.0+ Release
extern float __VERIFIER_nondet_float();
extern void __ESBMC_assume(_Bool);
extern void __ESBMC_assert(_Bool, const char*);

// Versao FLOAT32 do modelo matemático Llama.cpp / DeepSeek
float dot_product(const float* vec1, const float* vec2, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

int deepseek_moe_router(float* hidden_state) {
    float logits[NUM_EXPERTS];
    float max_logit = -1e9;
    int top1_expert = -1;

    for (int e = 0; e < NUM_EXPERTS; e++) {
        logits[e] = dot_product(hidden_state, router_weights[e], HIDDEN_DIM);
        
        if (logits[e] > max_logit) {
            max_logit = logits[e];
            top1_expert = e;
        }
    }
    
    return top1_expert; 
}

int main() {
    float base_token[HIDDEN_DIM];
    float adv_token[HIDDEN_DIM];

    // INJECAO MATEMATICO CONTINUA - PONTO FLUTUANTE
    for (int i = 0; i < HIDDEN_DIM; i++) {
        base_token[i] = __VERIFIER_nondet_float();
        __ESBMC_assume(base_token[i] >= -1.0f && base_token[i] <= 1.0f);
        adv_token[i] = base_token[i]; 
    }

    int expert_original = deepseek_moe_router(base_token);

    // ATAQUE ADVERSARIAL FLUTUANTE (EPSILON NOISE)
    float epsilon = __VERIFIER_nondet_float();
    __ESBMC_assume(epsilon >= -0.05f && epsilon <= 0.05f); 
    adv_token[0] += epsilon; 

    int expert_atacado = deepseek_moe_router(adv_token);

    // ASSERCAO 
    __ESBMC_assert(expert_original == expert_atacado, 
                   "VULNERABILIDADE FLOAT32: Roteador sensivel a perturbacao harmonica!");

    return 0;
}
