import os
import argparse
import random

def generate_mock_weights(output_path, num_experts=4, hidden_dim=8):
    """
    Gera pesos fakes em PONTO FLUTUANTE.
    O objetivo agora é forçar os solvers robustos (Boolector/MathSAT) a provarem 
    a matemática contínua real, ignorando soluções de contorno em inteiros.
    """
    print(f"Gerando matriz MOCK em FLOAT32 para o roteador [{num_experts}x{hidden_dim}]...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("// ========================================================\n")
        f.write("// PESOS DE EXTRAÇÃO FLOAT32: DEEPSEEK ROUTER MOE\n")
        f.write("// ========================================================\n\n")
        f.write(f"#define HIDDEN_DIM {hidden_dim}\n")
        f.write(f"#define NUM_EXPERTS {num_experts}\n\n")
        
        f.write("const float router_weights[NUM_EXPERTS][HIDDEN_DIM] = {\n")
        for i in range(num_experts):
            f.write("    {")
            # Simulando distribuição float
            row = [round(random.uniform(-0.1, 0.1), 6) for _ in range(hidden_dim)]
            f.write(", ".join(map(str, row)))
            if i == num_experts - 1:
                f.write("}\n")
            else:
                f.write("},\n")
        f.write("};\n")
    print(f"Sucesso! Cabeçalho FLOAT32 C gravado em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--out", type=str, default="../include/router_weights.h")
    
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(base_dir, args.out)

    if args.mock:
        generate_mock_weights(out_file)
