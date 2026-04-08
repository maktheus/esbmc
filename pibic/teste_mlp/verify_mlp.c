#include <stdio.h>
#include <math.h>
#include "mlp_weights.h"

// ESBMC intrinsics
void __VERIFIER_assert(int cond) {
    if (!(cond)) {
        // ESBMC will detect this as a property violation
        int *p = 0;
        *p = 0;
    }
}

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Logic: sigmoid(x) > 0.5 iff x > 0
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
    // We formally verify the 4 possible discrete inputs for XOR
    // (0,0) -> 0
    __VERIFIER_assert(mlp_forward_score(0.0f, 0.0f) <= 0.0f);
    
    // (1,1) -> 0
    __VERIFIER_assert(mlp_forward_score(1.0f, 1.0f) <= 0.0f);
    
    // (0,1) -> 1
    __VERIFIER_assert(mlp_forward_score(0.0f, 1.0f) > 0.0f);
    
    // (1,0) -> 1
    __VERIFIER_assert(mlp_forward_score(1.0f, 0.0f) > 0.0f);
    
    return 0;
}
