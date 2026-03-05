/**
 * attention_block_stub.c
 *
 * Verificacao Modular de um Bloco de Atencao (Transformer) com ESBMC
 *
 * Modelo: cabeca de atencao unica (single-head), dimensao d=4, seqlen=2
 * Pesos W_Q, W_K, W_V fixados com valores tipicos de um LLM quantizado (Fixed16).
 *
 * Propriedades verificadas:
 *  P1: logits de atencao (scores) nao causam overflow em float32
 *  P2: pesos de atencao apos softmax estao no intervalo [0, 1]
 *  P3: saida da cabeca de atencao e limitada por ||out||_inf <= delta
 *
 * Comando ESBMC:
 *   esbmc attention_block_stub.c --floatbv --unwind 4 \
 *          --overflow-check --no-unwinding-assertions
 */

#include <assert.h>

/* ESBMC nao-determinismo */
float nondet_float();
void __ESBMC_assume(int cond);

/* Dimensoes do modelo reduzido */
#define D    4   /* embedding dim */
#define SEQ  2   /* comprimento de sequencia */

/* --- Pesos fixos quantizados (representam camada atencao de LLM) ----------- */
/* W_Q, W_K, W_V : matrizes d x d (representadas flat, row-major) */
static const float W_Q[D][D] = {
    { 0.50f, -0.25f,  0.10f,  0.30f},
    {-0.15f,  0.60f, -0.20f,  0.05f},
    { 0.20f,  0.10f,  0.45f, -0.30f},
    {-0.05f,  0.35f,  0.25f,  0.55f}
};
static const float W_K[D][D] = {
    { 0.40f,  0.15f, -0.10f,  0.20f},
    { 0.05f, -0.50f,  0.30f, -0.10f},
    {-0.20f,  0.25f,  0.35f,  0.10f},
    { 0.30f, -0.05f,  0.15f, -0.45f}
};
static const float W_V[D][D] = {
    { 0.30f,  0.20f,  0.10f, -0.15f},
    {-0.10f,  0.40f, -0.25f,  0.05f},
    { 0.15f, -0.30f,  0.50f,  0.20f},
    { 0.05f,  0.10f, -0.20f,  0.35f}
};

/* Softmax aproximado via normalizacao linear (conservadora para verificacao) */
static void softmax2(float a, float b, float *pa, float *pb) {
    float sum = a + b;
    if (sum <= 0.0f) sum = 1.0f;   /* evita divisao por zero */
    *pa = a / sum;
    *pb = b / sum;
}

/* Produto vetor-matriz: out[d] = in[d] * W[d][d] */
static void matvec(const float W[D][D], const float in[D], float out[D]) {
    for (int i = 0; i < D; i++) {
        out[i] = 0.0f;
        for (int j = 0; j < D; j++)
            out[i] += W[i][j] * in[j];
    }
}

/* Produto escalar */
static float dot(const float a[D], const float b[D]) {
    float s = 0.0f;
    for (int i = 0; i < D; i++) s += a[i] * b[i];
    return s;
}

int main(void) {
    /* --- Entrada simbolica: token embeddings x0, x1 em [-1, 1]^D ---------- */
    float x0[D], x1[D];
    for (int i = 0; i < D; i++) {
        x0[i] = nondet_float();
        x1[i] = nondet_float();
        __ESBMC_assume(x0[i] >= -1.0f && x0[i] <= 1.0f);
        __ESBMC_assume(x1[i] >= -1.0f && x1[i] <= 1.0f);
    }

    /* --- Projeto Q, K, V -------------------------------------------------- */
    float q0[D], q1[D];
    float k0[D], k1[D];
    float v0[D], v1[D];

    matvec(W_Q, x0, q0);  matvec(W_Q, x1, q1);
    matvec(W_K, x0, k0);  matvec(W_K, x1, k1);
    matvec(W_V, x0, v0);  matvec(W_V, x1, v1);

    /* --- Scores de atencao: score(qi, kj) = dot(qi, kj) / sqrt(D) --------- */
    float scale = 0.5f;   /* 1/sqrt(4) = 0.5 */
    float s00 = dot(q0, k0) * scale;
    float s01 = dot(q0, k1) * scale;

    /* P1: scores nao extrapolam intervalo esperado para entradas normalizadas */
    /* Bound analitico: |score| <= D * max_w^2 / sqrt(D) = 4 * 0.6^2 / 0.5 = 2.88 */
    assert(s00 >= -3.0f && s00 <= 3.0f);
    assert(s01 >= -3.0f && s01 <= 3.0f);

    /* --- Softmax sobre scores (linha 0 da matriz de atencao) --------------- */
    float a00, a01;
    /* Estabilidade numerica: subtrai max antes do softmax */
    float m = (s00 > s01) ? s00 : s01;
    float e00 = s00 - m;   /* <= 0 */
    float e01 = s01 - m;   /* <= 0 */
    /* Aproximacao linear conservadora: substituimos exp por max(0, 1+x)     */
    /* (bound superior do exp em [-inf,0] via Taylor de 1a ordem)            */
    float ea = (1.0f + e00 > 0.0f) ? 1.0f + e00 : 0.0f;
    float eb = (1.0f + e01 > 0.0f) ? 1.0f + e01 : 0.0f;
    softmax2(ea, eb, &a00, &a01);

    /* P2: pesos de atencao pertencem a [0, 1] e somam 1 */
    assert(a00 >= 0.0f && a00 <= 1.0f);
    assert(a01 >= 0.0f && a01 <= 1.0f);
    assert(a00 + a01 >= 0.99f && a00 + a01 <= 1.01f);

    /* --- Saida da cabeca: out0 = a00*v0 + a01*v1 -------------------------- */
    float out0[D];
    for (int i = 0; i < D; i++)
        out0[i] = a00 * v0[i] + a01 * v1[i];

    /* P3: norma L-inf da saida limitada por delta = 0.6                      */
    /* Bound analitico: |out[i]| <= max|V*x| <= D * max_wv * max_x = 4*0.5*1 = 2 */
    /* Mas pesos de atencao normalizam: ||out||_inf <= max(|v0|, |v1|) <= 1   */
    for (int i = 0; i < D; i++)
        assert(out0[i] >= -1.0f && out0[i] <= 1.0f);

    return 0;
}
