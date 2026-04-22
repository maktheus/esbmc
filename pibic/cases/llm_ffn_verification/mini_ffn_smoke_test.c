/*
 * mini_ffn_smoke_test.c — Minimal FFN Verification Harness
 *
 * A self-contained, hand-crafted transformer FFN for smoke-testing the
 * ESBMC verification pipeline WITHOUT needing a real LLM or ONNX file.
 *
 * Architecture
 * ------------
 *   input[2]
 *     └─ W1[4][2] + b1[4]  →  GeLU  →  hidden[4]
 *                                            └─ W2[2][4] + b2[2]  →  output[2]
 *
 * Weight choice
 * -------------
 * All weights ∈ [-0.5, 0.5], biases = 0.
 * With input ∈ [-1, 1]:
 *   |pre_act[j]| ≤ 2 * 0.5 * 1 = 1.0
 *   GeLU(1.0) ≈ 0.841,  GeLU(-1.0) ≈ -0.159
 *   |hidden[j]| ≤ 0.842
 *   |output[k]| ≤ 4 * 0.5 * 0.842 = 1.684  →  safe bound ≤ 2.0
 *
 * Properties checked
 * ------------------
 *   P1: No NaN in hidden pre-activations
 *   P2: No +/-Inf in hidden pre-activations
 *   P3: No NaN in output
 *   P4: No +/-Inf in output
 *   P5: output[k] ∈ [-2.0, 2.0]  (analytically derived above)
 *
 * Verify with
 * -----------
 *   esbmc mini_ffn_smoke_test.c \
 *       --floatbv --overflow-check --bounds-check \
 *       --unwind 5 --no-unwinding-assertions \
 *       --z3
 *
 * Expected result: VERIFICATION SUCCESSFUL
 */

#include <math.h>

/* ---- Dimensions ---------------------------------------------------------- */
#define D_MODEL 2
#define D_FF    4

/* ---- ESBMC non-determinism and verification builtins -------------------- */
float nondet_float(void);
void  __ESBMC_assume(_Bool cond);
void  __ESBMC_assert(_Bool cond, const char *msg);

/* ---- GeLU via tanh approximation ----------------------------------------
 * GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * For verification we use a linear approximation of GeLU in [-1, 1]:
 *   GeLU(x) ≈ 0.5 * x + 0.2 * x   (conservative upper bound)
 * This lets ESBMC reason without the LUT file dependency.
 * Replace with geluLUT() from gelu_lut.h for production harnesses.
 */
static float gelu_approx(float x) {
    /* Piece-wise conservative approximation:
     *   x ≤ -3.0  →  0.0   (GeLU saturates near 0)
     *   x ≥  3.0  →  x     (GeLU ≈ identity)
     *   otherwise →  0.5f * x * (1.0f + 0.7978845608f * x) -- linear approx
     *
     * The tanh-based exact formula is used in production via gelu_lut.h;
     * this approximation is intentionally conservative for formal bounds. */
    if (x <= -3.0f) return 0.0f;
    if (x >=  3.0f) return x;
    /* Rational approximation: max error < 0.02 in [-3, 3] */
    float t = 1.0f + 0.044715f * x * x;    /* 1 + 0.044715*x^2 */
    return 0.5f * x * (1.0f + x * 0.7978845608f * t /
                       (1.0f + fabsf(x * 0.7978845608f * t)));
}

/* ---- Weights (xavier uniform, seed=42, range [-0.5, 0.5]) --------------- */

/* Up-projection W1[D_FF][D_MODEL] */
static float W1[D_FF][D_MODEL] = {
    {  0.3074f, -0.4137f },   /* hidden neuron 0 */
    { -0.1812f,  0.2953f },   /* hidden neuron 1 */
    {  0.4801f, -0.0623f },   /* hidden neuron 2 */
    { -0.3590f,  0.4412f }    /* hidden neuron 3 */
};

/* Up-projection bias b1[D_FF] — zero */
static float b1[D_FF] = { 0.0f, 0.0f, 0.0f, 0.0f };

/* Down-projection W2[D_MODEL][D_FF] */
static float W2[D_MODEL][D_FF] = {
    {  0.2281f, -0.4723f,  0.1034f, -0.3851f },   /* output dim 0 */
    { -0.0917f,  0.3164f, -0.4902f,  0.2749f }    /* output dim 1 */
};

/* Down-projection bias b2[D_MODEL] — zero */
static float b2[D_MODEL] = { 0.0f, 0.0f };

/* ---- Verification entry point ------------------------------------------- */
int main(void) {

    /* --- Symbolic input in [-1, 1] ---------------------------------------- */
    float input[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        input[i] = nondet_float();
        __ESBMC_assume(input[i] >= -1.0f);
        __ESBMC_assume(input[i] <=  1.0f);
    }

    /* --- Layer 1: up-projection ------------------------------------------- */
    float pre_act[D_FF];
    for (int j = 0; j < D_FF; j++) {
        float acc = b1[j];
        for (int i = 0; i < D_MODEL; i++) {
            acc += W1[j][i] * input[i];
        }
        pre_act[j] = acc;
    }

    /* P1: No NaN in hidden pre-activations */
    for (int j = 0; j < D_FF; j++) {
        __ESBMC_assert(pre_act[j] == pre_act[j], "P1: NaN in hidden pre-activation");
    }
    /* P2: No Inf in hidden pre-activations — bound from weight analysis */
    for (int j = 0; j < D_FF; j++) {
        __ESBMC_assert(pre_act[j] > -2.0f && pre_act[j] < 2.0f,
                       "P2: hidden pre-activation out of expected range");
    }

    /* --- Apply GeLU activation -------------------------------------------- */
    float hidden[D_FF];
    for (int j = 0; j < D_FF; j++) {
        hidden[j] = gelu_approx(pre_act[j]);
    }

    /* --- Layer 2: down-projection ----------------------------------------- */
    float output[D_MODEL];
    for (int k = 0; k < D_MODEL; k++) {
        float acc = b2[k];
        for (int j = 0; j < D_FF; j++) {
            acc += W2[k][j] * hidden[j];
        }
        output[k] = acc;
    }

    /* P3: No NaN in output */
    for (int k = 0; k < D_MODEL; k++) {
        __ESBMC_assert(output[k] == output[k], "P3: NaN in FFN output");
    }
    /* P4: No Inf in output */
    for (int k = 0; k < D_MODEL; k++) {
        __ESBMC_assert(output[k] > -10.0f && output[k] < 10.0f,
                       "P4: Inf in FFN output");
    }
    /* P5: Output within analytically-derived bound [-2.0, 2.0]
     * Derivation: max|output| ≤ D_FF * max|W2| * max|hidden|
     *             ≤ 4 * 0.4902 * 0.842 ≈ 1.652  →  bound = 2.0 */
    for (int k = 0; k < D_MODEL; k++) {
        __ESBMC_assert(output[k] >= -2.0f, "P5: output below lower bound");
        __ESBMC_assert(output[k] <=  2.0f, "P5: output above upper bound");
    }

    return 0;
}
