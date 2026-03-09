#include <stdlib.h>

void matmul_tiled(float* A, float* B, float* C, int M, int N, int K, int TILE) {
    for (int i = 0; i < M; i += TILE) {
        for (int j = 0; j < N; j += TILE) {
            for (int k = 0; k < K; ++k) {
                for (int ii = 0; ii < TILE; ++ii) {
                    for (int jj = 0; jj < TILE; ++jj) {
                        int row = i + ii;
                        int col = j + jj;
                        // BUG: Missing boundary checks leads to Out Of Bounds
                        C[row * N + col] += A[row * K + k] * B[k * N + col];
                    }
                }
            }
        }
    }
}

void verify_kernel() {
    int M = 2; int N = 2; int K = 2; int TILE = 2; // Exact multiples for faster ESBMC resolution
    float* A = (float*)calloc(M * K, sizeof(float));
    float* B = (float*)calloc(K * N, sizeof(float));
    float* C = (float*)calloc(M * N, sizeof(float));
    if (A && B && C) matmul_tiled(A, B, C, M, N, K, TILE);
    if(A) free(A);
    if(B) free(B);
    if(C) free(C);
}

int main() { verify_kernel(); return 0; }
