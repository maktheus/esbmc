#include <assert.h>

float nondet_float();

float relu(float x) {
    if (x > 0.0f) {
        return x;
    }
    return 0.0f;
}

void dense_layer(float inputs[2], float weights[2][2], float bias[2], float output[2]) {
    float val0 = (inputs[0] * weights[0][0]) + (inputs[1] * weights[0][1]) + bias[0];
    output[0] = relu(val0);
    
    float val1 = (inputs[0] * weights[1][0]) + (inputs[1] * weights[1][1]) + bias[1];
    output[1] = relu(val1);
}

float output_layer(float inputs[2], float weights[2], float bias) {
    float val = (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + bias;
    return val;
}

int main() {
    float in_x1 = nondet_float();
    float in_x2 = nondet_float();
    
    __ESBMC_assume(in_x1 >= 0.0f && in_x1 <= 1.0f);
    __ESBMC_assume(in_x2 >= 0.0f && in_x2 <= 1.0f);
    
    float inputs[2] = {in_x1, in_x2};
    
    float w_hidden[2][2] = {
        {0.5f, -0.2f},
        {-0.1f, 0.8f}
    };
    float b_hidden[2] = {0.1f, -0.05f};
    
    float hidden_out[2];
    dense_layer(inputs, w_hidden, b_hidden, hidden_out);
    
    float w_out[2] = {1.0f, 0.5f};
    float b_out = 0.0f;
    
    float final_score = output_layer(hidden_out, w_out, b_out);
    
    assert(final_score <= 1.0f);
    assert(final_score >= 0.0f);
    
    return 0;
}
