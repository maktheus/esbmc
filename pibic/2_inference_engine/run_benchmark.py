import subprocess
import time
import csv
import os

def run_benchmark():
    sizes = [2, 3, 4, 5, 6] # Verification grows exponentially, keep small for demo
    results_file = "pibic/results/case2_benchmark.csv"
    
    print("--- Starting Case 2 Benchmark (Scalability) ---")
    
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['MatrixSize', 'Time(s)', 'Result'])
        
        for size in sizes:
            print(f"Verifying Matrix Size {size}x{size}...")
            start_time = time.time()
            
            # Pass macro DIM_LIMIT to compiler
            # ESBMC passes -D flags to frontend
            cmd = [
                "esbmc", 
                "pibic/2_inference_engine/matmul_kernel.cpp",
                "--multi-property", 
                "--memory-leak-check", 
                "--overflow-check",
                "--smtlib",
                "--unwind", "10",
                "-DDIM_LIMIT=" + str(size)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                duration = time.time() - start_time
                success = "VERIFICATION SUCCESSFUL" in result.stdout
                outcome = "Pass" if success else "Fail"
            except subprocess.TimeoutExpired:
                duration = 120.0
                outcome = "Timeout"
            
            print(f"Size {size}: {outcome} in {duration:.2f}s")
            writer.writerow([size, f"{duration:.4f}", outcome])
            csvfile.flush()

if __name__ == "__main__":
    run_benchmark()
