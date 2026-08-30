import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core_verify.esbmc_caller import run_esbmc, UNSAFE
from core_verify.SMT_feedback_parser import FeedbackTrace

RL_C_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cases', 'ai_model_checking', 'rl_policy.c'))

def test_rl_actor_critic_bounds():
    """
    Tests if the ESBMC Python API can catch a Reinforcement Learning agent 
    outputting an invalid continuous action mathematically (e.g. steering > 1.0)
    through exact floatbv constraints.
    """
    result = run_esbmc(RL_C_FILE, z3=True, floatbv=True, unwind=1)
    
    assert result.status == UNSAFE, (
        "esperado contraexemplo da politica RL; obtido "
        f"status={result.status} rc={result.returncode}\n{result.output[-600:]}"
    )
    print("RL Policy Boundary Bug successfully caught by Formal Verification.")
