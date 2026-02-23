#include <assert.h>
#include "nn_forward_pass.c"

// ESBMC intrinsics for non-determinism
float nondet_float();

int main() {
    float input[INPUT_SIZE];
    
    // 1. Initialize inputs non-deterministically with constraints
    // Example: Input is bounded between -0.5 and 0.5
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = nondet_float();
        __ESBMC_assume(input[i] >= -0.5f && input[i] <= 0.5f);
    }
    
    // 2. Run the network
    run_network(input);
    
    // 3. Verify a property
    // Example Property: If inputs are small, output[0] should not be exceedingly large.
    // This is a dummy property. A real robustness property would be:
    // If input is close to X, output should be close to Y.
    
    // Let's assert that output[0] is within some range for all valid inputs.
    // If this assertion fails, ESBMC will provide a counter-example (input values) that violates it.
    // We pick a large bound to likely pass, or a small one to likely fail for demonstration.
    
    __ESBMC_assert(output[0] > 0.0f, "Output 0 should be greater than 0.0");
    
    return 0;
}
