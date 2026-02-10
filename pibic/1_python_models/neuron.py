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
