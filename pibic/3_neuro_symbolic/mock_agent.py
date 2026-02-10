import os
import subprocess
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
    print(f"[ESBMC] Verifying {filename}...")
    start_time = time.time()
    # Run ESBMC
    # --no-pointer-check to keep output simple for this demo
    cmd = ["esbmc", filename, "--overflow-check", "--memory-leak-check", "--no-pointer-check", "--smtlib"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    success = "VERIFICATION SUCCESSFUL" in result.stdout
    duration = end_time - start_time
    return success, result.stdout, duration

def main():
    print("--- Starting Neuro-Symbolic Agent Loop (Benchmark) ---")
    
    c_file = "generated_code.c"
    max_iterations = 5
    pibic_dir = "pibic/results"
    if not os.path.exists(pibic_dir):
        os.makedirs(pibic_dir)
        
    results_file = os.path.join(pibic_dir, "case3_agent_stats.csv")
    
    print(f"[Agent] optimized path: {results_file}")

    try:
        # Initialize CSV
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Iteration', 'Success', 'Duration(s)', 'CodeSize(bytes)'])
            
            for i in range(max_iterations):
                # 1. Generate/Refine Code
                code = call_llm("Fix the code", i)
                
                with open(c_file, "w") as f:
                    f.write(code)
                    
                print(f"[Agent] Wrote code to {c_file}")
                
                # 2. Verify
                success, output, duration = verify_code(c_file)
                
                # Log metrics
                writer.writerow([i, success, f"{duration:.4f}", len(code)])
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
