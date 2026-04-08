import tensorflow as tf
import numpy as np
import torch
from mlp_training import MLP

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load PyTorch model
model_pt = MLP()
model_pt.load_state_dict(torch.load("mlp_model.pth"))
model_pt.eval()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mlp_model_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Comparativo de Precisão: PyTorch vs LiteRT INT8")
print("-" * 50)

# Quantization parameters
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

for x_raw in [[0.0, 0.0], [0.1, 0.9], [0.9, 0.1], [1.0, 1.0]]:
    # PyTorch inference
    with torch.no_grad():
        y_pt = model_pt(torch.tensor([x_raw], dtype=torch.float32)).numpy()[0][0]
    
    # TFLite INT8 inference (requires scaling)
    x_quant = np.array([x_raw], dtype=np.float32) / input_scale + input_zero_point
    interpreter.set_tensor(input_details[0]['index'], x_quant.astype(np.int8))
    interpreter.invoke()
    y_quant = interpreter.get_tensor(output_details[0]['index'])[0]
    y_tflite = float((y_quant.astype(np.float32) - output_zero_point) * output_scale)
    
    print(f"Input: {x_raw}")
    print(f"  PyTorch: {y_pt:.4f}")
    print(f"  TFLite:  {y_tflite:.4f} (Raw INT8: {y_quant[0]})")
    print(f"  Erro:    {abs(y_pt - y_tflite):.4f}")
    print("-" * 30)
