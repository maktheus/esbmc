#include <iostream>
#include <vector>
#include <cmath> // Keep cmath for std::sqrt
#include <cassert> // Keep cassert if assert is used elsewhere, though not in this snippet
#include <cstdlib> // Keep cstdlib for malloc/free

// Mock of ESBMC intrinsics for standalone compilation
// __ESBMC_assume is handled natively by ESBMC
extern "C" int nondet_int();

// Simplified Dot-Product Attention mechanism simulation
// Logic: Calculate scores = Query * Key^T / sqrt(d_k)
float dot_product_attention(float* query, float* key, int seq_len) {
    float score = 0.0f;
    
    // POTENTIAL BUG 1: Missing bounds check on seq_len if not constrained
    for (int i = 0; i < seq_len; ++i) {
        // POTENTIAL BUG 2: buffer overflow if arrays are smaller than seq_len
        score += query[i] * key[i]; 
    }
    
    // Normalization
    if (seq_len > 0) {
        score /= std::sqrt((float)seq_len);
    }
    
    return score;
}

void verify_kernel() {
    // 1. Model Nondeterministic Inputs (Symbolic Execution)
    int seq_len = nondet_int();
    
    // 2. Pre-conditions / Constraints
    // We assume realistic sequence lengths for a small layer
    __ESBMC_assume(seq_len > 0);
    __ESBMC_assume(seq_len <= 10); 

    // 3. Allocate Memory
    float* query = (float*)malloc(seq_len * sizeof(float));
    float* key = (float*)malloc(seq_len * sizeof(float));

    // Check for malloc failure
    if (!query || !key) {
        free(query);
        free(key);
        return;
    }

    // Initialize with nondet values (abstracting actual weights)
    // For this test, we just care about memory safety, not values

    // 4. Run the Kernel
    float result = dot_product_attention(query, key, seq_len);

    // 5. Post-conditions
    // Result should be a valid number (not NaN, though harder to prove without initialized floats)
    // More importantly, ESBMC checks for buffer overflows during the loop above.
    
    // Cleanup
    free(query);
    free(key);
}

int main() {
    verify_kernel();
    return 0;
}
