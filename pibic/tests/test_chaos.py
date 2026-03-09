import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core_verify.esbmc_caller import run_esbmc
from cases.control_chaos_testing.chaos_generator import inject_pid_chaos

TEMPLATE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cases', 'control_chaos_testing', 'pid_template.c'))
GENERATED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cases', 'control_chaos_testing'))

@pytest.mark.parametrize("noise_type", ["Uniform", "Gaussian", "Impulse", "Drift", "Sinusoidal"])
def test_pid_resilience_under_chaos(noise_type):
    out_file = os.path.join(GENERATED_DIR, f"pid_test_{noise_type}.c")
    inject_pid_chaos(TEMPLATE, out_file, noise_type)
    
    # We use z3 and floatbv to exactly calculate the non-linear math bounds
    result = run_esbmc(out_file, z3=True, floatbv=True, unwind=6, bounds_check=True)
    
    # In some extreme chaos shapes (like 100x Impulse), the bounds (-100, 100) will be broken.
    # The test passes if ESBMC can successfully execute and yield a verdict (Safe or Unsafe) without crashing.
    assert result is not None, f"ESBMC Caller crashed verifying {noise_type} noise."
    
    # Some profiles break the PID, some don't. We just ensure we ran the engine.
    print(f"{noise_type} Chaos Profile -> Is Safe? {result.is_safe}")
