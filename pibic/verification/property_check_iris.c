#include <stdio.h>
#include "nn_forward_pass_iris.c"

// ESBMC intrinsics for non-determinism
float nondet_float();

int main() {
    float input[INPUT_SIZE];
    
    // Base input from Test Sample 0
    float base_input[INPUT_SIZE] = {0.31099753f, -0.59237301f, 0.53540856f, 0.00087755f};
    
    // Define a small perturbation bound e.g., epsilon = 0.01
    float epsilon = 0.01f;

    // Apply non-deterministic perturbations bounded by epsilon
    for (int i = 0; i < INPUT_SIZE; i++) {
        __ESBMC_assume(input[i] >= base_input[i] - epsilon && input[i] <= base_input[i] + epsilon);
    }
    
    // Run the network
    run_network(input);
    
    // The expected class for the base input is 1.
    // Assert that the perturbed input also predicts class 1 (local robustness)
    __ESBMC_assert(output[1] > output[0] && output[1] > output[2], "Prediction changes within epsilon neighborhood!");

    return 0;
}
