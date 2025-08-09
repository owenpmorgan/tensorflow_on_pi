
# First test at getting tensor flow working on RPi5

# RPi only has tflite, so first will pass on pi
# It will not be present on macOS so it will retrieve the lite syntax from the master tensorflow package
# must use venv in _Python/_python_venvs/tensorflow_mac311
# Python version 3.11 for compatibility

try:
    import tflite_runtime.interpreter as tflite
    backend = "tflite-runtime (likely Raspberry Pi)"
except ImportError:
    import tensorflow.lite as tflite
    backend = "full TensorFlow (likely macOS)"

print(f"[INFO] Using backend: {backend}")
print(f"[INFO] Interpreter class from: {tflite.Interpreter}")

# Optional: path to a .tflite model file for a real inference test
# For now, you can leave it None to just test that TF loads
model_path = None  # e.g., "hello_world.tflite"

if model_path:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"[DEBUG] Input details: {input_details}")
    print(f"[DEBUG] Output details: {output_details}")

    # Fill with dummy input data
    import numpy as np
    dummy_input = np.array([0.0], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"[RESULT] Output data: {output_data}")

else:
    print("[INFO] No model file provided, TensorFlow Lite backend is working.")