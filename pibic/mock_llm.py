import sys

task_id = sys.argv[1]
iteration = int(sys.argv[2])
file_path = sys.argv[3]

case_3_bad = r"""
#include <stdlib.h>
#include <string.h>

void parse_csv(char* input) {
    char buffer[10]; 
    strcpy(buffer, input); // BUG: Buffer overflow
}

int main() {
    char* input = calloc(20, 1);
    if(input) {
        parse_csv(input);
        free(input);
    }
    return 0;
}
"""

case_3_good = r"""
#include <stdlib.h>
#include <string.h>

void parse_csv(char* input) {
    char buffer[10];
    strncpy(buffer, input, 9); // FIX: Safe copy
    buffer[9] = '\0';
}

int main() {
    char* input = calloc(20, 1);
    if(input) {
        parse_csv(input);
        free(input);
    }
    return 0;
}
"""

case_2_bad = r"""
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
"""

case_2_good = r"""
#include <stdlib.h>

void matmul_tiled(float* A, float* B, float* C, int M, int N, int K, int TILE) {
    for (int i = 0; i < M; i += TILE) {
        for (int j = 0; j < N; j += TILE) {
            for (int k = 0; k < K; ++k) {
                for (int ii = 0; ii < TILE; ++ii) {
                    for (int jj = 0; jj < TILE; ++jj) {
                        int row = i + ii;
                        int col = j + jj;
                        // FIX: Boundary checks added
                        if (row < M && col < N) {
                            C[row * N + col] += A[row * K + k] * B[k * N + col];
                        }
                    }
                }
            }
        }
    }
}

void verify_kernel() {
    int M = 3; int N = 3; int K = 3; int TILE = 2; // Not multiples!
    float* A = (float*)calloc(M * K, sizeof(float));
    float* B = (float*)calloc(K * N, sizeof(float));
    float* C = (float*)calloc(M * N, sizeof(float));
    if (A && B && C) matmul_tiled(A, B, C, M, N, K, TILE);
    if(A) free(A);
    if(B) free(B);
    if(C) free(C);
}

int main() { verify_kernel(); return 0; }
"""

content = ""
if task_id == "CASE_3_BUFFER":
    content = case_3_bad if iteration == 1 else case_3_good
elif task_id == "CASE_2_MATMUL":
    content = case_2_bad if iteration <= 2 else case_2_good

with open(file_path, "w") as f:
    f.write(content.strip() + "\n")
