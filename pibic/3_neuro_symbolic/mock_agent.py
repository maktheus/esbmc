import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_verify.esbmc_caller import (
    PARSE_ERROR, SAFE, USAGE_ERROR, run_esbmc,
)
import time
import csv
import random

# Mock LLM response simulating a "fix" loop
# Iteration 0: Generates code with a buffer overflow
# Iteration 1: Generates fixed code
responses = [
    # Ref: Bad Code
    r"""
#include <stdlib.h>
#include <string.h>

void parse_csv(char* input) {
    // BUG: Fixed size buffer, input can be larger
    char buffer[10]; 
    strcpy(buffer, input);
}

int main() {
    char* input = malloc(20);
    // Abstract input
    parse_csv(input);
    free(input);
    return 0;
}
    """,
    # Ref: Good Code
    r"""
#include <stdlib.h>
#include <string.h>

void parse_csv(char* input) {
    // FIX: Dynamic allocation or bounds check
    // Here we just use strncpy for safety
    char buffer[10];
    strncpy(buffer, input, 9);
    buffer[9] = '\0';
}

int main() {
    char* input = malloc(20);
    parse_csv(input);
    free(input);
    return 0;
}
    """
]

def call_llm(prompt, iteration):
    print(f"\n[Agent] Asking LLM (Iteration {iteration})...")
    # Simulate LLM processing delay
    time.sleep(random.uniform(0.5, 2.0))
    return responses[min(iteration, len(responses)-1)]

def verify_code(filename):
    """Verifica o codigo gerado e devolve (sucesso, saida, duracao).

    DOIS DEFEITOS CORRIGIDOS, ambos suficientes para o loop nunca funcionar:

      1. `--smtlib` faz o ESBMC **emitir a formula SMT em vez de resolve-la**,
         entao "VERIFICATION SUCCESSFUL" nunca aparecia e `success` era
         permanentemente falso. O loop gastava as 5 iteracoes sem poder
         terminar, e `case3_agent_stats.csv` ficou com 0 bytes.
      2. Caminho fixo para `build/src/esbmc/esbmc`, que nao existe. Agora usa
         o `esbmc_caller`, que resolve o binario e distingue erro de execucao
         de propriedade violada.
    """
    print(f"[ESBMC] Verificando {os.path.basename(filename)}...")
    # --unwind e obrigatorio: o modelo de strncpy da biblioteca do ESBMC
    # desenrola >24000 iteracoes quando o tamanho da origem e desconhecido, e
    # a verificacao nunca termina. Sem bound, TODA iteracao dava timeout, e o
    # loop realimentava o LLM com um resultado INDECISO como se fosse
    # contraexemplo -- a mesma confusao que este projeto corrige em outros
    # pontos.
    # `--unwind 32` SEM `--no-unwinding-assertions`. A primeira tentativa de
    # correcao usou as duas juntas e tornou o caso VACUO: com o desenrolamento
    # cortado e a assercao suprimida, o `strcpy` estourado passava a reportar
    # SAFE. Medido: unwind=32 sozinho -> UNSAFE em 0,1 s; com
    # no_unwinding_assertions -> SAFE em 0,1 s. A flag que resolve um caso
    # (mascarar unwinding que esconde vazamento) destroi outro.
    r = run_esbmc(filename, timeout=120, unwind=32, overflow_check=True,
                  memory_leak_check=True, no_pointer_check=True)
    if not r.verificou:
        # indeciso NAO e "bug encontrado": nao ha contraexemplo a realimentar
        print(f"[ESBMC] SEM VEREDITO: {r.status} rc={r.returncode} "
              f"({r.time_taken:.1f}s)")
    return r.status == SAFE, r.output, r.time_taken, r.status

def main():
    print("--- Starting Neuro-Symbolic Agent Loop (Benchmark) ---")
    
    # caminhos ancorados no arquivo, nao no CWD: com "pibic/results" relativo,
    # rodar de dentro de 3_neuro_symbolic/ criava 3_neuro_symbolic/pibic/results/
    AQUI = os.path.dirname(os.path.abspath(__file__))
    PIBIC = os.path.dirname(AQUI)
    c_file = os.path.join(AQUI, "generated_code.c")
    max_iterations = 5
    results_dir = os.path.join(PIBIC, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "case3_agent_stats.csv")
    
    print(f"[Agent] metricas -> {results_file}")

    try:
        # Initialize CSV
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Iteration', 'Success', 'Status', 'Duration(s)', 'CodeSize(bytes)'])
            
            for i in range(max_iterations):
                # 1. Generate/Refine Code
                code = call_llm("Fix the code", i)
                
                with open(c_file, "w") as f:
                    f.write(code)
                    
                print(f"[Agent] Wrote code to {c_file}")
                
                # 2. Verify
                success, output, duration, status = verify_code(c_file)
                
                # Log metrics
                writer.writerow([i, success, status, f"{duration:.4f}", len(code)])
                csvfile.flush() # Ensure data is written
                
                if success:
                    print(f"\n[Success] Verified code passed all checks in iteration {i}!")
                    print(f"Time taken: {duration:.2f}s")
                    break
                else:
                    print(f"\n[Failure] Verification failed in iteration {i}.")
                    print(f"Time taken: {duration:.2f}s")
                    # print("ESBMC Output (Snippet):")
                    # print("\n".join(output.splitlines()[-5:]))
                    print("\n[Agent] Feeding counterexample back to LLM...")
    except Exception as e:
        print(f"[Fatal Error] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
