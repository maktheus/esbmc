import os
import subprocess
import time

# Resolve ESBMC binary path dynamically assuming script is in pibic/core_verify
ESBMC_BIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build", "src", "esbmc", "esbmc"))

class VerificationResult:
    def __init__(self, is_safe, stdout, stderr, time_taken, timeout_occurred=False):
        self.is_safe = is_safe
        self.stdout = stdout
        self.stderr = stderr
        self.time_taken = time_taken
        self.timeout_occurred = timeout_occurred

def build_esbmc_cmd(filepath, **kwargs):
    """
    Constructs the ESBMC CLI command safely handling flags specified in PRD.
    """
    cmd = [ESBMC_BIN, filepath]
    
    # 1. Solvers
    if kwargs.get('z3'): cmd.append('--z3')
    if kwargs.get('bitwuzla'): cmd.append('--bitwuzla')
    if kwargs.get('mathsat'): cmd.append('--mathsat')
    if kwargs.get('cvc4'): cmd.append('--cvc4')
    
    # 2. Encoding Types
    if kwargs.get('floatbv'): cmd.append('--floatbv')
    if kwargs.get('fixedbv'): cmd.append('--fixedbv')
    
    # 3. Loops and Assertions Setup
    if kwargs.get('unwind') is not None:
        cmd.extend(['--unwind', str(kwargs.get('unwind'))])
    if kwargs.get('no_unwinding_assertions'): cmd.append('--no-unwinding-assertions')
    if kwargs.get('k_induction'): cmd.append('--k-induction')
    
    # 4. Built-in property checkers (Bug Checks)
    if kwargs.get('memory_leak_check'): cmd.append('--memory-leak-check')
    if kwargs.get('overflow_check'): cmd.append('--overflow-check')
    if kwargs.get('bounds_check'): cmd.append('--bounds-check')
    if kwargs.get('pointer_check'): cmd.append('--pointer-check')
    
    # 5. Pipeline and API configs
    if kwargs.get('multi_property'): cmd.append('--multi-property')
    if kwargs.get('smt_formula_only'): cmd.append('--smt-formula-only')
    
    return cmd

def run_esbmc(filepath, timeout=60, **kwargs):
    """
    Robust Subprocess wrapper mapping Python parameters to native ESBMC C++ calls.
    Returns a structured VerificationResult preventing generic bash/python deadlocks.
    """
    if not os.path.exists(ESBMC_BIN):
        raise FileNotFoundError(f"ESBMC binary missing at: {ESBMC_BIN}")
        
    cmd = build_esbmc_cmd(filepath, **kwargs)
    
    start_time = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        time_taken = time.time() - start_time
        
        # Consider STDOUT and STDERR because solvers format traces differently
        is_safe = "VERIFICATION SUCCESSFUL" in proc.stdout or "VERIFICATION SUCCESSFUL" in proc.stderr
        return VerificationResult(is_safe, proc.stdout, proc.stderr, time_taken)
        
    except subprocess.TimeoutExpired as e:
        time_taken = time.time() - start_time
        outs = e.stdout.decode('utf-8') if e.stdout else "TIMEOUT EXPIRED (STDOUT TRUNCATED)"
        errs = e.stderr.decode('utf-8') if e.stderr else "TIMEOUT EXPIRED (STDERR TRUNCATED)"
        return VerificationResult(False, outs, errs, time_taken, timeout_occurred=True)
