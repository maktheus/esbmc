#include <iostream>
#include <vector>
#include <cstdlib>

// ESBMC intrinsics are handled natively


// Tiled Matrix Multiplication (Simplified)
// C = A * B
// A: M x K, B: K x N, C: M x N
void matmul_tiled(float* A, float* B, float* C, int M, int N, int K, int TILE_SIZE) {
    
    // Iterate over tiles
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            
            // Iterate inside tile
            // BUG POTENTIAL: What if M, N are not multiples of TILE_SIZE?
            // Need boundary checks: (i+ii < M) and (j+jj < N)
            
            for (int k = 0; k < K; ++k) {
                // Optimize memory access patterns (naive here for verification)
                
                for (int ii = 0; ii < TILE_SIZE; ++ii) {
                    for (int jj = 0; jj < TILE_SIZE; ++jj) {
                        
                        int row = i + ii;
                        int col = j + jj;
                        
                        if (row < M && col < N) {
                            // C[row * N + col] += A[row * K + k] * B[k * N + col];
                            
                            // Pointer arithmetic verification
                            int c_idx = row * N + col;
                            int a_idx = row * K + k;
                            int b_idx = k * N + col;
                            
                            // Check bounds implicitly by access
                            // ESBMC effectively checks:
                            // 0 <= c_idx < M*N
                            // 0 <= a_idx < M*K
                            // 0 <= b_idx < K*N
                            
                            C[c_idx] += A[a_idx] * B[b_idx];
                        }
                    }
                }
            }
        }
    }
}

void verify_kernel() {
    int M = nondet_int();
    int N = nondet_int();
    int K = nondet_int();
    int TILE_SIZE = 4; // Check small tile size
    
    // Constraints
#ifdef DIM_LIMIT
    __ESBMC_assume(M == DIM_LIMIT);
    __ESBMC_assume(N == DIM_LIMIT);
    __ESBMC_assume(K == DIM_LIMIT);
#else
    __ESBMC_assume(M > 0 && M <= 8);
    __ESBMC_assume(N > 0 && N <= 8);
    __ESBMC_assume(K > 0 && K <= 8);
#endif
    
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));
    
    if (A && B && C) {
        matmul_tiled(A, B, C, M, N, K, TILE_SIZE);
    }
    
    free(A);
    free(B);
    free(C);
}

int main() {
    verify_kernel();
    return 0;
}
