# Case 3: Neuro-Symbolic Verification (Agentic Loop)

This example simulates an "Agentic Loop" where an LLM generates C code, and ESBMC automatically verifies it. If verification fails, the agent would theoretically use the error trace to prompt the LLM to fix the code.

## Files

- `mock_agent.py`: A Python script that mocks this loop. It cycles through a predefined list of "LLM responses"—starting with buggy code and progressing to fixed code—verifying each step with ESBMC.

## How to Run

```bash
python3 mock_agent.py
```

## Expected Flow

1. **Iteration 0**: The "LLM" provides code with `strcpy` into a fixed-size buffer. ESBMC detects an overflow.
2. **Iteration 1**: The "LLM" provides fixed code using `strncpy`. ESBMC returns `VERIFICATION SUCCESSFUL`.
