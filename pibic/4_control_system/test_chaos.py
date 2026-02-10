import os
import subprocess
import time

def run_esbmc_chaos_test():
    print("--- Running Case 4: Control System Chaos Verification ---")
    
    c_file = "pibic/4_control_system/pid_controller.c"
    if not os.path.exists(c_file):
        print(f"Error: Could not find {c_file}")
        return

    # ESBMC Command
    # --floatbv: Enable floating-point bit-vector solver
    # --k-induction: Use k-induction for unbounded loops (though we have a bounded loop here, good practice)
    # --unwind 10: Unroll the loop 10 times matches the loop in C
    cmd = ["esbmc", c_file, "--floatbv", "--unwind", "11", "--no-unwinding-assertions", "--smtlib"]
    
    print(f"Executing: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        print(f"Verification Check Complete in {duration:.2f}s")
        
        if "VERIFICATION SUCCESSFUL" in result.stdout:
            print("[SUCCESS] The Control System is ROBUST against the injected Chaos.")
            print("Property: Temperature never exceeds MAX_SAFE_TEMP even with sensor noise.")
        else:
            print("[FAILURE] Chaos injection caused a safety violation!")
            print("Counterexample found. The system is unstable under specific noise conditions.")
            # Print last few lines of output to see the failure
            print("\n".join(result.stdout.splitlines()[-20:]))
            
    except FileNotFoundError:
        print("Error: ESBMC not found. Please ensure it is in your PATH.")

if __name__ == "__main__":
    run_esbmc_chaos_test()
