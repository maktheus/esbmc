#include <stdio.h>
#include <math.h> 
#include "nn_weights.h"

// ReLU activation function
float relu(float x) {
    return (x > 0) ? x : 0;
}

// Global output array
float output[OUTPUT_SIZE];

// Forward pass function
void run_network(float input[INPUT_SIZE]) {
    float hidden[HIDDEN_SIZE];
    
    // Layer 1: Input -> Hidden
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = b1[i]; // Bias
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += input[j] * w1[i][j];
        }
        hidden[i] = relu(hidden[i]); // Activation
    }
    
    // Layer 2: Hidden -> Output
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = b2[i]; // Bias
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += hidden[j] * w2[i][j];
        }
        // No activation (or Linear) for the last layer in this example
    }
}
