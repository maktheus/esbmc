import re

scale_factor = 256 # 8-bit fractional quantization

# Extracting numbers from the known weights and manually recreating them
# Based on the earlier output of mlp_weights.h
w_hidden = [
    [2.334265, 2.773769],
    [-2.030052, 2.040186],
    [-0.146728, -1.516269],
    [-2.188007, -2.145728]
]
b_hidden = [-2.334243, -0.0101266, 2.018354, 2.187911]
w_out = [-2.689387, 2.604033, 2.410956, -5.160296]
b_out = 0.8777598

def quantize(val):
    return int(round(val * scale_factor))

qw_hidden = [[quantize(v) for v in row] for row in w_hidden]
qb_hidden = [quantize(v) for v in b_hidden]
qw_out = [quantize(v) for v in w_out]
qb_out = quantize(b_out)

c_code = f"""
#include <stdio.h>

extern void __VERIFIER_assume(int cond);
extern void __VERIFIER_assert(int cond) {{
    if (!cond) {{
        int *p = 0;
        *p = 0;
    }}
}}

// Constantes Inteiras Quantizadas (Scale: {scale_factor})
int qw_hidden[4][2] = {{
    {{{qw_hidden[0][0]}, {qw_hidden[0][1]}}},
    {{{qw_hidden[1][0]}, {qw_hidden[1][1]}}},
    {{{qw_hidden[2][0]}, {qw_hidden[2][1]}}},
    {{{qw_hidden[3][0]}, {qw_hidden[3][1]}}}
}};
int qb_hidden[4] = {{{qb_hidden[0]}, {qb_hidden[1]}, {qb_hidden[2]}, {qb_hidden[3]}}};
int qw_out[4] = {{{qw_out[0]}, {qw_out[1]}, {qw_out[2]}, {qw_out[3]}}};
int qb_out = {qb_out};

// Simulação Frama-C: Limites observados estatisticamente no Python (já na escala {scale_factor})
int MIN_HIDDEN_VAL = -{int(5.0 * scale_factor)}; // max value could be around 5.0
int MAX_HIDDEN_VAL = {int(5.0 * scale_factor)};

int relu_int(int x) {{
    return x > 0 ? x : 0;
}}

int mlp_forward_qnn(int x1, int x2) {{
    int hidden_outputs[4];
    for (int i = 0; i < 4; i++) {{
        // Matemática Ponto Fixo (Inteiros puros, amigável para ESP32)
        int h_val = (x1 * qw_hidden[i][0]) / {scale_factor} + (x2 * qw_hidden[i][1]) / {scale_factor} + qb_hidden[i];
        
        // --- INJEÇÃO DE INVARIANTE (Frama-C) ---
        // Podamos o espaço de busca matemático do ESBMC confirmando que a soma interna não colapsou
        __VERIFIER_assume(h_val >= MIN_HIDDEN_VAL && h_val <= MAX_HIDDEN_VAL);
        
        hidden_outputs[i] = relu_int(h_val);
    }}
    
    int score = qb_out;
    for (int i = 0; i < 4; i++) {{
        score += (hidden_outputs[i] * qw_out[i]) / {scale_factor};
    }}
    
    return score;
}}

int main() {{
    // Em Ponto Fixo (Escala {scale_factor}):
    // 0.0 é 0
    // 1.0 é {scale_factor}
    
    int zero = 0;
    int um = {scale_factor};
    
    // Testes XOR Reconstruídos
    __VERIFIER_assert(mlp_forward_qnn(zero, zero) <= 0);
    __VERIFIER_assert(mlp_forward_qnn(um, um) <= 0);
    __VERIFIER_assert(mlp_forward_qnn(zero, um) > 0);
    __VERIFIER_assert(mlp_forward_qnn(um, zero) > 0);
    
    return 0;
}}
"""

with open('verify_mlp_qnn.c', 'w') as f:
    f.write(c_code)

print("Gerado verify_mlp_qnn.c!")
