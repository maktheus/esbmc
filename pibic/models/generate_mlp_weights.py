import random
import os

# Configuration
INPUT_SIZE = 4      # Small input for testing
HIDDEN_SIZE = 8     # Small hidden layer
OUTPUT_SIZE = 2     # 2 classes
HEADER_FILE_PATH = "verification/nn_weights.h"

def generate_random_weights(rows, cols):
    """Generates a list of lists representing a matrix of random weights."""
    return [[random.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]

def generate_random_bias(size):
    """Generates a list of random biases."""
    return [random.uniform(-0.1, 0.1) for _ in range(size)]

def array_to_c_string(data):
    """Converts a 1D list to a C array string."""
    return "{" + ", ".join(f"{x:.6f}f" for x in data) + "}"

def matrix_to_c_string(data):
    """Converts a 2D list to a C multidimensional array string."""
    rows = []
    for row in data:
        rows.append(array_to_c_string(row))
    return "{\n    " + ",\n    ".join(rows) + "\n}"

def main():
    print(f"Generating weights for MLP: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}")
    
    w1 = generate_random_weights(HIDDEN_SIZE, INPUT_SIZE)
    b1 = generate_random_bias(HIDDEN_SIZE)
    
    w2 = generate_random_weights(OUTPUT_SIZE, HIDDEN_SIZE)
    b2 = generate_random_bias(OUTPUT_SIZE)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(HEADER_FILE_PATH), exist_ok=True)
    
    with open(HEADER_FILE_PATH, "w") as f:
        f.write("#ifndef NN_WEIGHTS_H\n")
        f.write("#define NN_WEIGHTS_H\n\n")
        
        f.write(f"#define INPUT_SIZE {INPUT_SIZE}\n")
        f.write(f"#define HIDDEN_SIZE {HIDDEN_SIZE}\n")
        f.write(f"#define OUTPUT_SIZE {OUTPUT_SIZE}\n\n")
        
        f.write(f"const float w1[HIDDEN_SIZE][INPUT_SIZE] = {matrix_to_c_string(w1)};\n\n")
        f.write(f"const float b1[HIDDEN_SIZE] = {array_to_c_string(b1)};\n\n")
        
        f.write(f"const float w2[OUTPUT_SIZE][HIDDEN_SIZE] = {matrix_to_c_string(w2)};\n\n")
        f.write(f"const float b2[OUTPUT_SIZE] = {array_to_c_string(b2)};\n\n")
        
        f.write("#endif // NN_WEIGHTS_H\n")
        
    print(f"Weights saved to {HEADER_FILE_PATH}")

if __name__ == "__main__":
    main()
