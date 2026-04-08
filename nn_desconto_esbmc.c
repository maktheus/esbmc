#include <assert.h>

// Funções primitivas para injeção do SMT Solver (Z3/MathSAT)
extern float __VERIFIER_nondet_float();
extern void __ESBMC_assume(int);
extern void __ESBMC_assert(int, const char*);

// A clássica função de ativação não-linear de uma Rede Neural
float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// ======================================================================
// OS PESOS DA IA (O equivalente exato ao seu `model_data.h` do Llama)
// Em um caso real, você exportaria esses números do PyTorch após treinar.
// Arquitetura: 1 Entrada (Qtd) -> 2 Neurônios (Ocultos) -> 1 Saída (Desconto)
// ======================================================================
const float W1[2] = {0.05f, 0.01f}; // Matriz de Pesos da Camada 1
const float b1[2] = {-0.5f, 0.0f};  // Vetor de Viés (Bias) da Camada 1

const float W2[2] = {0.06f, 0.02f}; // Matriz de Pesos da Camada de Saída
const float b2 = 0.00f;             // Viés Final

// ======================================================================
// O FORWARD PASS (Exatamente igual ao forward do Transformer, mas minúsculo)
// ======================================================================
float forward_neural_network(float entrada_quantidade) {
    
    // Matemática da primeira camada (Multiplicação de Matriz + Bias -> ReLU)
    float hidden[2];
    hidden[0] = relu((entrada_quantidade * W1[0]) + b1[0]);
    hidden[1] = relu((entrada_quantidade * W1[1]) + b1[1]);

    // Matemática da Saída Final (Produto Escalar -> Linear)
    float saida_desconto = (hidden[0] * W2[0]) + (hidden[1] * W2[1]) + b2;

    return saida_desconto;
}

int main() {
    // 1. O NONDET_FLOAT COMO ENTRADA (Simula todas as compras do mundo real)
    float quantidade = __VERIFIER_nondet_float();
    
    // Restringimos a regra de negócio da loja: compras de 0 a 100 peças.
    __ESBMC_assume(quantidade >= 0.0f && quantidade <= 100.0f);

    // 2. INFERÊNCIA DA REDE NEURAL NATIVA
    float desconto = forward_neural_network(quantidade);

    // 3. A PROPRIEDADE DE VERIFICAÇÃO (SMT SOLVER ASSERTION)
    // Queremos ter a comprovação matemática de que essa Rede Neural treinada
    // NUNCA passará de 30% de desconto sob NENHUMA HIPÓTESE.
    __ESBMC_assert(desconto <= 0.30f, "FALHA ESTRUTURAL DA REDE NEURAL: Desconto gerado foi maior que 30%!");

    return 0;
}
