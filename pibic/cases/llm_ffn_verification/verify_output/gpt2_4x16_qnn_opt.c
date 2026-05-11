/*
 * gpt2_4x16_qnn_opt.c — Optimized pure-abstract QNN harness (4×16, GPT-2)
 *
 * Same mathematical content as gpt2_4x16_qnn.c but stripped of dead code:
 *   - No input variables (decoupled from hidden in abstract mode)
 *   - No Layer-1 pre-activation computation
 *   - int32_t for hidden and accumulation (values bounded << 2^31)
 *   - fxp_mult uses 32-bit shift to match Q8.8 semantics exactly
 *
 * Weights, biases, and interval bounds are identical to gpt2_4x16_qnn.c.
 *
 * Verify:
 *   esbmc gpt2_4x16_qnn_opt.c --no-unwinding-assertions --z3
 *
 * Expected: VERIFICATION SUCCESSFUL
 */

#include <stdint.h>

/* 32-bit FXP Q8.8 — hidden values fit in [-60, 60], products fit in int32 */
static int32_t fxp32_mult(int32_t a, int32_t b) { return (a * b) >> 8; }

int  nondet_int(void);
void __ESBMC_assume(_Bool cond);
void __ESBMC_assert(_Bool cond, const char *msg);

/* W2[4][16] in Q8.8 (int32 literals) */
static int32_t W2_0[16] = { -20, 22, 21, 1, 7, -15, -3, -16, -25, 25, -13, 5, -10, -7, -5, -9 };
static int32_t W2_1[16] = { -24, 2, -23, 23, 19, -15, 10, 4, 16, -1, -16, 7, -22, 4, -15, 10 };
static int32_t W2_2[16] = { -20, 15, 23, -21, -18, 25, -16, -22, 24, 21, 8, 3, 22, 8, 8, 18 };
static int32_t W2_3[16] = { -25, -13, 6, -18, 4, -7, 2, -26, 18, -7, -19, -3, -1, 7, 1, -24 };

int main(void) {

    /* Hidden neurons: symbolic, bounded by analytical GELU interval */
    int32_t h[16];
    h[0]  = nondet_int(); __ESBMC_assume(h[0]  >= -22 && h[0]  <= 29);
    h[1]  = nondet_int(); __ESBMC_assume(h[1]  >= -12 && h[1]  <= 14);
    h[2]  = nondet_int(); __ESBMC_assume(h[2]  >= -28 && h[2]  <= 43);
    h[3]  = nondet_int(); __ESBMC_assume(h[3]  >= -31 && h[3]  <= 48);
    h[4]  = nondet_int(); __ESBMC_assume(h[4]  >= -29 && h[4]  <= 43);
    h[5]  = nondet_int(); __ESBMC_assume(h[5]  >= -17 && h[5]  <= 20);
    h[6]  = nondet_int(); __ESBMC_assume(h[6]  >= -29 && h[6]  <= 44);
    h[7]  = nondet_int(); __ESBMC_assume(h[7]  >= -31 && h[7]  <= 49);
    h[8]  = nondet_int(); __ESBMC_assume(h[8]  >= -27 && h[8]  <= 39);
    h[9]  = nondet_int(); __ESBMC_assume(h[9]  >= -24 && h[9]  <= 34);
    h[10] = nondet_int(); __ESBMC_assume(h[10] >= -29 && h[10] <= 43);
    h[11] = nondet_int(); __ESBMC_assume(h[11] >= -21 && h[11] <= 28);
    h[12] = nondet_int(); __ESBMC_assume(h[12] >= -21 && h[12] <= 27);
    h[13] = nondet_int(); __ESBMC_assume(h[13] >= -15 && h[13] <= 18);
    h[14] = nondet_int(); __ESBMC_assume(h[14] >= -21 && h[14] <= 28);
    h[15] = nondet_int(); __ESBMC_assume(h[15] >= -21 && h[15] <= 28);

    /* Layer 2: down-projection in Q8.8 (32-bit accumulation) */
    int32_t out0 = 0;
    out0 += fxp32_mult(W2_0[0],  h[0]);
    out0 += fxp32_mult(W2_0[1],  h[1]);
    out0 += fxp32_mult(W2_0[2],  h[2]);
    out0 += fxp32_mult(W2_0[3],  h[3]);
    out0 += fxp32_mult(W2_0[4],  h[4]);
    out0 += fxp32_mult(W2_0[5],  h[5]);
    out0 += fxp32_mult(W2_0[6],  h[6]);
    out0 += fxp32_mult(W2_0[7],  h[7]);
    out0 += fxp32_mult(W2_0[8],  h[8]);
    out0 += fxp32_mult(W2_0[9],  h[9]);
    out0 += fxp32_mult(W2_0[10], h[10]);
    out0 += fxp32_mult(W2_0[11], h[11]);
    out0 += fxp32_mult(W2_0[12], h[12]);
    out0 += fxp32_mult(W2_0[13], h[13]);
    out0 += fxp32_mult(W2_0[14], h[14]);
    out0 += fxp32_mult(W2_0[15], h[15]);

    int32_t out1 = 0;
    out1 += fxp32_mult(W2_1[0],  h[0]);
    out1 += fxp32_mult(W2_1[1],  h[1]);
    out1 += fxp32_mult(W2_1[2],  h[2]);
    out1 += fxp32_mult(W2_1[3],  h[3]);
    out1 += fxp32_mult(W2_1[4],  h[4]);
    out1 += fxp32_mult(W2_1[5],  h[5]);
    out1 += fxp32_mult(W2_1[6],  h[6]);
    out1 += fxp32_mult(W2_1[7],  h[7]);
    out1 += fxp32_mult(W2_1[8],  h[8]);
    out1 += fxp32_mult(W2_1[9],  h[9]);
    out1 += fxp32_mult(W2_1[10], h[10]);
    out1 += fxp32_mult(W2_1[11], h[11]);
    out1 += fxp32_mult(W2_1[12], h[12]);
    out1 += fxp32_mult(W2_1[13], h[13]);
    out1 += fxp32_mult(W2_1[14], h[14]);
    out1 += fxp32_mult(W2_1[15], h[15]);

    int32_t out2 = 0;
    out2 += fxp32_mult(W2_2[0],  h[0]);
    out2 += fxp32_mult(W2_2[1],  h[1]);
    out2 += fxp32_mult(W2_2[2],  h[2]);
    out2 += fxp32_mult(W2_2[3],  h[3]);
    out2 += fxp32_mult(W2_2[4],  h[4]);
    out2 += fxp32_mult(W2_2[5],  h[5]);
    out2 += fxp32_mult(W2_2[6],  h[6]);
    out2 += fxp32_mult(W2_2[7],  h[7]);
    out2 += fxp32_mult(W2_2[8],  h[8]);
    out2 += fxp32_mult(W2_2[9],  h[9]);
    out2 += fxp32_mult(W2_2[10], h[10]);
    out2 += fxp32_mult(W2_2[11], h[11]);
    out2 += fxp32_mult(W2_2[12], h[12]);
    out2 += fxp32_mult(W2_2[13], h[13]);
    out2 += fxp32_mult(W2_2[14], h[14]);
    out2 += fxp32_mult(W2_2[15], h[15]);

    int32_t out3 = 0;
    out3 += fxp32_mult(W2_3[0],  h[0]);
    out3 += fxp32_mult(W2_3[1],  h[1]);
    out3 += fxp32_mult(W2_3[2],  h[2]);
    out3 += fxp32_mult(W2_3[3],  h[3]);
    out3 += fxp32_mult(W2_3[4],  h[4]);
    out3 += fxp32_mult(W2_3[5],  h[5]);
    out3 += fxp32_mult(W2_3[6],  h[6]);
    out3 += fxp32_mult(W2_3[7],  h[7]);
    out3 += fxp32_mult(W2_3[8],  h[8]);
    out3 += fxp32_mult(W2_3[9],  h[9]);
    out3 += fxp32_mult(W2_3[10], h[10]);
    out3 += fxp32_mult(W2_3[11], h[11]);
    out3 += fxp32_mult(W2_3[12], h[12]);
    out3 += fxp32_mult(W2_3[13], h[13]);
    out3 += fxp32_mult(W2_3[14], h[14]);
    out3 += fxp32_mult(W2_3[15], h[15]);

    /* P1: output within analytically-derived bound ±56 (Q8.8) */
    __ESBMC_assert(out0 >= -56 && out0 <= 56, "P1: output[0] out of bound");
    __ESBMC_assert(out1 >= -56 && out1 <= 56, "P1: output[1] out of bound");
    __ESBMC_assert(out2 >= -56 && out2 <= 56, "P1: output[2] out of bound");
    __ESBMC_assert(out3 >= -56 && out3 <= 56, "P1: output[3] out of bound");

    return 0;
}
