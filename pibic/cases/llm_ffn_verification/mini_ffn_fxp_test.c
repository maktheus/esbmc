/*
 * mini_ffn_fxp_test.c — FFN em aritmética de ponto fixo Q8.8
 *
 * Demonstra que a verificação formal é TRATÁVEL quando:
 *   1. Pesos convertidos para INT16 (fixed-point Q8.8)
 *   2. ESBMC usa aritmética inteira (--fixedbv ou sem --floatbv)
 *   3. Dimensões pequenas: d_model=2, d_ff=4
 *
 * Formato Q8.8: o valor real v é representado como int16_t com SCALE=256
 *   float_to_fxp(v) = (int16_t)(v * 256)
 *   fxp_to_float(x) = x / 256.0
 *   fxp_mult(a, b)  = (a * b) >> 8      (rescala após multiplicação)
 *   fxp_add(a, b)   = a + b
 *
 * Rede verificada (mesmos pesos do mini_ffn_smoke_test.c):
 *   input[2]
 *     └─ W1[4][2] + b1[4] → ReLU → hidden[4]   (ReLU trivial em fxp: max(x,0))
 *          └─ W2[2][4] + b2[2] → output[2]
 *
 * Bounds analíticos em fxp (SCALE=256, input ∈ [-256, 256] ≡ float [-1, 1]):
 *   max |pre_act| ≤ 2 * 82 * 256 + 0 = 41984  (em int16, cabe em int32)
 *   max |hidden|  ≤ 41984  (ReLU não amplia)
 *   max |output|  ≤ 4 * 124 * (41984>>8) + 0 = 4 * 124 * 164 ≈ 81344
 *   Bound seguro para output: [-90000, 90000] em int32
 *
 * Verificar com:
 *   esbmc mini_ffn_fxp_test.c --overflow-check --unwind 5 \
 *         --no-unwinding-assertions --z3
 *
 * Resultado esperado: VERIFICATION SUCCESSFUL
 */

#include <stdint.h>

/* ---- Dimensões ---------------------------------------------------------- */
#define D_MODEL 2
#define D_FF    4
#define SCALE   256    /* 2^8 — bits fracionários */

/* ---- ESBMC builtins ------------------------------------------------------ */
int nondet_int(void);
void __ESBMC_assume(_Bool cond);
void __ESBMC_assert(_Bool cond, const char *msg);

/* ---- Aritmética fixed-point Q8.8 ----------------------------------------
 * Multiplicação: (a * b) >> 8  — produto intermediário em int32 */
static int32_t fxp_mult(int32_t a, int32_t b) {
    return (a * b) >> 8;
}

/* ---- Pesos em Q8.8 (float * 256, arredondado)
 * Original float:  W1[j][i] ∈ [-0.5, 0.5]
 * Q8.8:            W1[j][i] * 256 ∈ [-128, 128]                         */

/* W1[D_FF][D_MODEL] */
static int16_t W1[D_FF][D_MODEL] = {
    {  79, -106 },   /*  0.3074 * 256 ≈  79,  -0.4137 * 256 ≈ -106 */
    { -46,   76 },   /* -0.1812 * 256 ≈ -46,   0.2953 * 256 ≈   76 */
    { 123,  -16 },   /*  0.4801 * 256 ≈ 123,  -0.0623 * 256 ≈  -16 */
    { -92,  113 }    /* -0.3590 * 256 ≈ -92,   0.4412 * 256 ≈  113 */
};
static int16_t b1[D_FF] = { 0, 0, 0, 0 };

/* W2[D_MODEL][D_FF] */
static int16_t W2[D_MODEL][D_FF] = {
    {  58, -121,   26,  -99 },
    { -23,   81, -125,   70 }
};
static int16_t b2[D_MODEL] = { 0, 0 };

/* ---- Entry point --------------------------------------------------------- */
int main(void) {

    /* Symbolic input in Q8.8: [-256, 256] ≡ float [-1, 1] */
    int32_t input[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        input[i] = nondet_int();
        __ESBMC_assume(input[i] >= -256 && input[i] <= 256);
    }

    /* --- Layer 1: dot product + ReLU ------------------------------------- */
    int32_t pre_act[D_FF];
    for (int j = 0; j < D_FF; j++) {
        int32_t acc = (int32_t)b1[j];
        for (int i = 0; i < D_MODEL; i++) {
            acc += fxp_mult((int32_t)W1[j][i], input[i]);
        }
        pre_act[j] = acc;
    }

    /* P1: pre-activations within expected range
     * max: 2 * 123 * 256 >> 8 = 2 * 123 = 246. With two inputs: 492  */
    for (int j = 0; j < D_FF; j++) {
        __ESBMC_assert(pre_act[j] >= -512 && pre_act[j] <= 512,
                       "P1: hidden pre-act out of range");
    }

    /* ReLU in fixed-point: max(x, 0) */
    int32_t hidden[D_FF];
    for (int j = 0; j < D_FF; j++) {
        hidden[j] = pre_act[j] > 0 ? pre_act[j] : 0;
    }

    /* --- Layer 2: dot product ------------------------------------------- */
    int32_t output[D_MODEL];
    for (int k = 0; k < D_MODEL; k++) {
        int32_t acc = (int32_t)b2[k];
        for (int j = 0; j < D_FF; j++) {
            acc += fxp_mult((int32_t)W2[k][j], hidden[j]);
        }
        output[k] = acc;
    }

    /* P2: output within safe bound
     * max: 4 * 125 * 512 >> 8 = 4 * 125 * 2 = 1000                     */
    for (int k = 0; k < D_MODEL; k++) {
        __ESBMC_assert(output[k] >= -1200 && output[k] <= 1200,
                       "P2: output out of bound");
    }

    /* P3: output is non-NaN (always true for integers — compiler guard)   */
    for (int k = 0; k < D_MODEL; k++) {
        __ESBMC_assert(output[k] == output[k], "P3: NaN impossible for int");
    }

    return 0;
}
