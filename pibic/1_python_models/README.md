# Case 1: Direct Verification of Python Models

This example demonstrates how to use the experimental Python frontend in ESBMC to verify properties of a simple neural network component (a single neuron with ReLU activation).

## Files

- `neuron.py`: A Python script defining a neuron and asserting properties about its output given constrained inputs.

## How to Run

This requires ESBMC to be built with `-DENABLE_PYTHON_FRONTEND=ON`.

```bash
esbmc neuron.py --floatbv --k-induction
```

## Expected Result

ESBMC should prove that the assertions hold (output is non-negative and bounded) for all possible input values within the range [0.0, 1.0].
