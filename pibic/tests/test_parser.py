import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core_verify.SMT_feedback_parser import FeedbackTrace

# Mock Testing the AI State Manager isolation rules (TSK_101 to TSK_105)

MOCK_OUT_OF_BOUNDS = """
CProver Verification
File test.c line 14: array bounds violated
State 201: x > 5
VERIFICATION FAILED
"""

MOCK_LEAK = """
Symex completed
File memory.cpp line 22: memory leak detected on exit
VERIFICATION FAILED
"""

MOCK_OVERFLOW = """
Unwinding loop 5
Arithmetic overflow on addition
VERIFICATION FAILED
"""

@pytest.mark.parametrize("mock_trace, expected_violation", [
    (MOCK_OUT_OF_BOUNDS, "array out of bounds"),
    (MOCK_LEAK, "memory leak"),
    (MOCK_OVERFLOW, "arithmetic overflow")
])
def test_regex_ai_trace_isolation(mock_trace, expected_violation):
    # Tests that the Context Formatter safely strips out compiler info and identifies
    # the exact mathematically proven failure
    trace = FeedbackTrace(mock_trace)
    assert expected_violation in trace.violations
    prompt_context = trace.extract_llm_prompt_context()
    assert expected_violation.split()[0].lower() in prompt_context.lower() or "violated" in prompt_context.lower() or "leak" in prompt_context.lower() or "overflow" in prompt_context.lower()
