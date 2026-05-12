/*
 * verify_mlp_qnn.c — Harness de verificacao formal (ESBMC)
 *
 * Gerado por quantize_mlp.py a partir de mlp_model.onnx
 *
 * Modelo: MLP 2->4->1, XOR
 * Quantizacao: inteiros puros, scale=256 (Q8.8)
 *
 * Propriedades verificadas (4 casos concretos XOR):
 *   P1: mlp(0,0) <= 0   (XOR = false)
 *   P2: mlp(1,1) <= 0   (XOR = false)
 *   P3: mlp(0,1)  > 0   (XOR = true)
 *   P4: mlp(1,0)  > 0   (XOR = true)
 *
 * Verificar:
 *   esbmc verify_mlp_qnn.c --no-unwinding-assertions --boolector
 *
 * Esperado: VERIFICATION SUCCESSFUL
 */

#include <stdint.h>

void __ESBMC_assume(_Bool cond);
void __ESBMC_assert(_Bool cond, const char *msg);

/* Pesos quantizados — scale=256 */
static int qw_hidden[4][2] = {
    {-183, -182},
    {-750, 750},
    {-555, -555},
    {-153, 952}
};
static int qb_hidden[4] = {181, 0, 555, 153};
static int qw_out[4]    = {-154, 1064, -888, -639};
static int qb_out       = 1037;

static int relu_int(int x) { return x > 0 ? x : 0; }

static int mlp_forward(int x1, int x2) {
    int h[4], i;
    for (i = 0; i < 4; i++) {
        int pre = (x1 * qw_hidden[i][0]) / 256
                + (x2 * qw_hidden[i][1]) / 256
                + qb_hidden[i];
        __ESBMC_assume(pre >= -1280 && pre <= 1280);
        h[i] = relu_int(pre);
    }
    int score = qb_out;
    for (i = 0; i < 4; i++)
        score += (h[i] * qw_out[i]) / 256;
    return score;
}

int main(void) {
    int zero = 0, um = 256;

    /* P1 */ __ESBMC_assert(mlp_forward(zero, zero) <= 0, "P1: XOR(0,0) deve ser false");
    /* P2 */ __ESBMC_assert(mlp_forward(um,   um  ) <= 0, "P2: XOR(1,1) deve ser false");
    /* P3 */ __ESBMC_assert(mlp_forward(zero, um  )  > 0, "P3: XOR(0,1) deve ser true");
    /* P4 */ __ESBMC_assert(mlp_forward(um,   zero)  > 0, "P4: XOR(1,0) deve ser true");

    return 0;
}
