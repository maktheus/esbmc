#include <stdio.h>
#include <math.h>
#include "mlp_weights.h"

extern void __VERIFIER_assert(int cond) {
    if (!(cond)) {
        int *p = 0;
        *p = 0;
    }
}

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float mlp_forward_score(float x1, float x2) {
    float hidden_outputs[4];
    for (int i = 0; i < 4; i++) {
        hidden_outputs[i] = relu(x1 * w_hidden[i][0] + x2 * w_hidden[i][1] + b_hidden[i]);
    }
    
    float score = b_out;
    for (int i = 0; i < 4; i++) {
        score += hidden_outputs[i] * w_out[i];
    }
    
    return score;
}

int main() {
    // Single point verification: (1, 1) -> 0
    float x1 = 1.0f;
    float x2 = 1.0f;
    float score = mlp_forward_score(x1, x2);
    __VERIFIER_assert(score <= 0.0f);
    return 0;
}
