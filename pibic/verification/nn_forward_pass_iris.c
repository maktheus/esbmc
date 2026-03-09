#include "nn_weights_iris.h"
#include <math.h>

float output[OUTPUT_SIZE];

void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0.0f) {
            x[i] = 0.0f;
        }
    }
}

void run_network(float input[INPUT_SIZE]) {
    float hidden_layer[HIDDEN_SIZE];

    // Hidden layer
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_layer[i] = b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden_layer[i] += w1[i][j] * input[j];
        }
    }
    relu(hidden_layer, HIDDEN_SIZE);

    // Output layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += w2[i][j] * hidden_layer[j];
        }
    }
}
