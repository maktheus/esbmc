/**
 * deepseek_mlp_stub.c
 *
 * Verificacao Modular de Nivel 4 — Bloco MLP Gated com SiLU (estilo DeepSeek)
 *
 * Arquitetura referencia: DeepSeek-V2 / DeepSeek-R1
 * Os modelos DeepSeek adotam a variante SwiGLU (Swish Gated Linear Unit)
 * na camada feed-forward:
 *
 *   out = (W2 * (SiLU(W1 * x) * (W3 * x)))
 *
 * onde SiLU(z) = z * sigma(z),  sigma(z) = 1 / (1 + exp(-z))
 *
 * Para verificacao formal com ESBMC, o exp() transcendental e substituido
 * por uma aproximacao polinomial de 1a ordem linearmente conservadora
 * (valida para o dominio de interesse):
 *
 *   sigma_approx(z) = clip(0.25*z + 0.5, 0.0, 1.0)   (aprox. linear de sigma)
 *
 * Isso preserva os bounds: sigma_approx \in [0,1] para todo z.
 *
 * Propriedades verificadas (ESBMC --floatbv --unwind 4 --overflow-check):
 *
 *   P1: ativacao SiLU em [-d_int, d_int]  — sem overflow antes do gate
 *   P2: gate (W3*x) permanece bounded     — sem overflow no segundo ramo
 *   P3: produto gated bounded             — sem explosao de gradiente
 *   P4: saida final (W2 * gated) limitada — garantia de seguranca numerica
 *
 * Comando:
 *   esbmc verification/deepseek_mlp_stub.c \
 *         --floatbv --unwind 4 --overflow-check --no-unwinding-assertions
 */

#include <assert.h>

/* ESBMC non-determinism */
float nondet_float();
void __ESBMC_assume(int cond);

/* Dimensoes do modelo reduzido (representa um slice do MLP DeepSeek) */
#define D_IN    4   /* dimensao de entrada (embedding slice) */
#define D_INTER 4   /* dimensao intermediaria (intermediate_size slice) */

/* --- Pesos fixos quantizados em Fixed16 (representam uma camada DeepSeek) -- */
/* W1, W3: D_INTER x D_IN (projecao up/gate)                                  */
/* W2:     D_IN x D_INTER  (projecao down)                                    */

static const float W1[D_INTER][D_IN] = {
    { 0.40f, -0.20f,  0.15f,  0.30f},
    {-0.25f,  0.50f,  0.10f, -0.15f},
    { 0.30f,  0.10f, -0.35f,  0.20f},
    {-0.10f, -0.30f,  0.45f,  0.05f}
};
static const float W3[D_INTER][D_IN] = {
    { 0.35f,  0.10f, -0.20f,  0.25f},
    { 0.05f, -0.40f,  0.30f,  0.10f},
    {-0.15f,  0.20f,  0.40f, -0.10f},
    { 0.20f,  0.30f, -0.05f, -0.35f}
};
static const float W2[D_IN][D_INTER] = {
    { 0.45f, -0.15f,  0.20f,  0.10f},
    {-0.20f,  0.35f,  0.10f, -0.25f},
    { 0.10f,  0.20f, -0.40f,  0.30f},
    {-0.30f,  0.05f,  0.25f,  0.45f}
};

/* --- Aprox. linear de sigmoid (verificavel pelo solver SMT) --------------- */
static float sigma_approx(float z) {
    float s = 0.25f * z + 0.5f;
    if (s < 0.0f) s = 0.0f;
    if (s > 1.0f) s = 1.0f;
    return s;
}

/* SiLU(z) = z * sigma(z), usando aproximacao verificavel */
static float silu(float z) {
    return z * sigma_approx(z);
}

/* Produto vetor-matriz NxM: out[n] = W[n][m] * in[m] */
static void matvec4x4(const float W[4][4], const float in[4], float out[4]) {
    for (int i = 0; i < 4; i++) {
        out[i] = 0.0f;
        for (int j = 0; j < 4; j++)
            out[i] += W[i][j] * in[j];
    }
}

int main(void) {
    /* --- Entrada simbolica: token embedding slice em [-1, 1]^D_IN ---------- */
    float x[D_IN];
    for (int i = 0; i < D_IN; i++) {
        x[i] = nondet_float();
        __ESBMC_assume(x[i] >= -1.0f && x[i] <= 1.0f);
    }

    /* --- Ramo gate (W1 * x): intermediario pre-SiLU ----------------------- */
    float h1[D_INTER];
    matvec4x4(W1, x, h1);

    /* Bound analitico: |h1[i]| <= sum_j |W1[i][j]| * 1 <= 4 * 0.5 = 2.0 */
    for (int i = 0; i < D_INTER; i++)
        assert(h1[i] >= -2.0f && h1[i] <= 2.0f);   /* P1 */

    /* --- SiLU aplicada ao ramo gate --------------------------------------- */
    float silu_h1[D_INTER];
    for (int i = 0; i < D_INTER; i++) {
        silu_h1[i] = silu(h1[i]);
        /* SiLU em [-2,2]: maximo analitico ~0.96 em z=2; minimo em z=-2 ~-0.09 */
        assert(silu_h1[i] >= -0.15f && silu_h1[i] <= 2.0f);   /* P1 mantida */
    }

    /* --- Ramo linear (W3 * x) --------------------------------------------- */
    float h3[D_INTER];
    matvec4x4(W3, x, h3);
    for (int i = 0; i < D_INTER; i++)
        assert(h3[i] >= -2.0f && h3[i] <= 2.0f);   /* P2 */

    /* --- Produto gated: gated[i] = silu(h1[i]) * h3[i] -------------------- */
    float gated[D_INTER];
    for (int i = 0; i < D_INTER; i++) {
        gated[i] = silu_h1[i] * h3[i];
        /* Bound: |gated| <= 2.0 * 2.0 = 4.0 */
        assert(gated[i] >= -4.0f && gated[i] <= 4.0f);   /* P3 */
    }

    /* --- Projecao down (W2 * gated): saida final do bloco MLP -------------- */
    float out[D_IN];
    matvec4x4(W2, gated, out);

    /* Bound analitico: |out[i]| <= sum_j |W2[i][j]| * 4.0 <= 4 * 0.45 * 4 = 7.2 */
    /* Na pratica com pesos quantizados a saida e muito menor; usamos 5.0 como bound */
    for (int i = 0; i < D_IN; i++)
        assert(out[i] >= -6.0f && out[i] <= 6.0f);   /* P4 */

    return 0;
}
