# Applying ESBMC to Generative AI: A Practical Guide

This document details three distinct approaches to leveraging ESBMC (Efficient SMT-Based Context-Bounded Model Checker) in the context of Generative AI.

## Approach 1: Direct Verification of AI Models (Python Frontend)

This approach treats the AI model code (written in Python) as the target for verification. It is useful for verifying architectural properties, data flow safety, and simple logical constraints of small models or specific layers.

### **Step-by-Step Implementation**

1. **Rebuild ESBMC with Python Support**:
    The default build does not include the experimental Python frontend. You must rebuild ESBMC to enable it.

    ```bash
    cd ~/esbmc/build
    rm -rf *
    cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=/usr/local -DENABLE_PYTHON_FRONTEND=ON
    cmake --build .
    sudo cmake --install .
    ```

2. **Define the Model in Python**:
    Create a Python file (e.g., `model_check.py`) representing your neural network layer or logic. You must use type hints (`float`, `int`) for ESBMC to infer types correctly.

    ```python
    # model_check.py
    def relu(x: float) -> float:
        if x > 0:
            return x
        return 0.0

    def neuron(input_a: float, input_b: float) -> float:
        # Weights (fixed for verification or symbolic)
        w1: float = 0.5
        w2: float = -0.2
        bias: float = 0.1
        
        # Linear combination
        activation: float = (input_a * w1) + (input_b * w2) + bias
        return relu(activation)

    def main() -> None:
        # Symbolic inputs (ESBMC treats uninitialized variables as non-deterministic/symbolic)
        in1: float
        in2: float
        
        # Constrain inputs (Pre-conditions)
        if in1 >= 0.0 and in1 <= 1.0 and in2 >= 0.0 and in2 <= 1.0:
            out: float = neuron(in1, in2)
            
            # Verify property (Post-condition)
            # Example: Output should never be negative (ReLU property)
            assert out >= 0.0
            
            # Example: Output should not exceed a theoretical max bounds
            # Max possible = (1.0 * 0.5) + (0.0 * -0.2) + 0.1 = 0.6
            assert out <= 0.6

    main()
    ```

3. **Run ESBMC**:

    ```bash
    esbmc model_check.py --floatbv --k-induction
    ```

---

## Approach 2: Verification of the Inference Engine (C/C++)

Generative AI models run on high-performance engines written in C++ (e.g., `llama.cpp`, `ONNX Runtime`). Bugs in these engines can lead to crashes, security vulnerabilities, or incorrect calculations.

### **Step-by-Step Implementation**

1. **Target a Runtime Component**:
    Select a specific C++ file or function from an inference engine (e.g., the matrix multiplication kernel or tokenization logic in `llama.cpp`).

2. **Prepare the Verification Harness**:
    Create a standalone C++ file that imports the target function and sets up the verification environment.

    ```cpp
    // verify_kernel.cpp
    #include "llama_util.h" // Hypothetical header
    #include <assert.h>

    void verify_attention_mechanism() {
        // Nondeterministic inputs
        int seq_len = nondet_int();
        __ESBMC_assume(seq_len > 0 && seq_len < 1024); // Constrain input size
        
        float* query = malloc(seq_len * sizeof(float));
        float* key = malloc(seq_len * sizeof(float));
        
        // Check for buffer overflows or invalid memory access
        if (query && key) {
             float score = dot_product_attention(query, key, seq_len);
             assert(!isnan(score)); // Check for NaN output
        }
    }
    ```

3. **Run ESBMC**:

    ```bash
    esbmc verify_kernel.cpp --overflow-check --memory-leak-check
    ```

---

## Approach 3: Neuro-Symbolic Verification (Agentic Loop)

This involves using ESBMC to verify code *generated* by an LLM. This creates a feedback loop where the LLM writes code, ESBMC checks it, and if a bug is found, the error trace is fed back to the LLM to fix the code.

### **Step-by-Step Implementation**

1. **Setup the Agent**:
    Create a script (Python) that interacts with an LLM API (OpenAI, Anthropic, or local Ollama).

2. **The Loop**:
    * **Prompt**: "Write a C function to parse a CSV string safely."
    * **LLM Output**: Generates `parse_csv.c`.
    * **Verification**: The script automatically runs:

        ```bash
        esbmc parse_csv.c --memory-leak-check --overflow-check
        ```

    * **Analysis**:
        * If `VERIFICATION SUCCESSFUL`: Code is accepted.
        * If `VERIFICATION FAILED`: Capture the "Counterexample" trace from ESBMC output.
    * **Refinement**: Send the counterexample back to the LLM: "Your code failed verification. Here is the error trace showing a buffer overflow when the input string is empty. Please fix it."

3. **Execution**:
    This approach turns ESBMC into a "super-linter" for AI-generated code, ensuring high reliability for synthesized software.
