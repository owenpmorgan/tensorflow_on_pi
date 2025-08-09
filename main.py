
# First test at getting tensor flow working on RPi5

# RPi only has tflite, so first will pass on pi
# It will not be present on macOS so it will retrieve the lite syntax from the master tensorflow package
# must use venv in _Python/_python_venvs/tensorflow_mac311
# Python version 3.11 for compatibility

# First test at getting TensorFlow working on RPi5 and macOS

# Detect platform backend
try:
    import tflite_runtime.interpreter as tflite
    backend = "tflite-runtime (likely Raspberry Pi)"
except ImportError:
    import tensorflow.lite as tflite
    backend = "full TensorFlow (likely macOS)"

print(f"[INFO] Using backend: {backend}")
print(f"[INFO] Interpreter class from: {tflite.Interpreter}")

# Path to .tflite model file
model_path = "hello_world.tflite"

import numpy as np

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"[DEBUG] Input details: {input_details}")
print(f"[DEBUG] Output details: {output_details}")

# Match the model's expected shape exactly (usually [1, 1] for hello_world.tflite)
expected_shape = input_details[0]['shape']
dummy_input = np.zeros(expected_shape, dtype=np.float32)
dummy_input[0][0] = 0.5  # Example input value

interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"[RESULT] Output data: {output_data}")
