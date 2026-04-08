import tensorflow as tf
import numpy as np
import onnx2tf

# 1. Convert ONNX to TensorFlow (SavedModel)
onnx2tf.convert(
    input_onnx_file_path="mlp_model.onnx",
    output_folder_path="mlp_tf_saved_model",
    copy_onnx_input_output_names_to_tflite=True,
    non_verbose=True
)

# 2. Convert TF SavedModel to TFLite with Post-Training Quantization (INT8)
converter = tf.lite.TFLiteConverter.from_saved_model("mlp_tf_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    # Provide sample data for quantization calibration (XOR inputs)
    for data in [
        [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]
    ]:
        yield [np.array([data], dtype=np.float32)]

converter.representative_dataset = representative_dataset
# Ensure that if ops can't be quantized, it fails or uses float (fallback)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# 3. Save the model
with open("mlp_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print("Model quantized and saved to mlp_model_quantized.tflite")
