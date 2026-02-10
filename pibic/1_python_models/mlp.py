from typing import List

def relu(x: float) -> float:
    if x > 0.0:
        return x
    return 0.0

def dense_layer(inputs: List[float], weights: List[List[float]], bias: List[float]) -> List[float]:
    output: List[float] = []
    # Hardcoded loop for 2 neurons in hidden layer to simpler verification
    # Neuron 0
    val0: float = (inputs[0] * weights[0][0]) + (inputs[1] * weights[0][1]) + bias[0]
    output.append(relu(val0))
    
    # Neuron 1
    val1: float = (inputs[0] * weights[1][0]) + (inputs[1] * weights[1][1]) + bias[1]
    output.append(relu(val1))
    
    return output

def output_layer(inputs: List[float], weights: List[float], bias: float) -> float:
    # Single neuron output
    val: float = (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + bias
    # Linear output (regression) or Sigmoid (classification)
    # Here we use linear for bounds checking
    return val

def main() -> None:
    # Input layer (2 features)
    in_x1: float
    in_x2: float
    
    # Preconditions: Normalized inputs
    if in_x1 >= 0.0 and in_x1 <= 1.0 and in_x2 >= 0.0 and in_x2 <= 1.0:
        inputs: List[float] = [in_x1, in_x2]
        
        # Hidden Layer Weights (2 inputs -> 2 neurons)
        # w[neuron_idx][input_idx]
        w_hidden: List[List[float]] = [
            [0.5, -0.2], # Neuron 0 weights
            [-0.1, 0.8]  # Neuron 1 weights
        ]
        b_hidden: List[float] = [0.1, -0.05]
        
        hidden_out: List[float] = dense_layer(inputs, w_hidden, b_hidden)
        
        # Output Layer Weights (2 inputs from hidden -> 1 output)
        w_out: List[float] = [1.0, 0.5]
        b_out: float = 0.0
        
        final_score: float = output_layer(hidden_out, w_out, b_out)
        
        # Verification Properties
        
        # Prop 1: Output is bounded given bounded inputs and weights
        # Max Hidden 0: (1.0*0.5 + 0.0) + 0.1 = 0.6 -> ReLU -> 0.6
        # Max Hidden 1: (0.0 + 1.0*0.8) - 0.05 = 0.75 -> ReLU -> 0.75
        # Max Final: (0.6 * 1.0) + (0.75 * 0.5) + 0.0 = 0.6 + 0.375 = 0.975
        
        assert final_score <= 1.0
        assert final_score >= -1.0 # Should be >= -0.05 actually? No, relu outputs >=0.
        # Hidden outputs are >= 0.
        # Weights are positive. Output should be >= 0?
        # w_out has positive weights. b_out is 0.
        # So final_score should be >= 0.
        assert final_score >= 0.0

main()
