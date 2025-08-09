# RPi only has tflite, so first will pass on pi
# It will not be present on macOS so it will retrieve the lite syntax from the master tensorflow package
# must use venv in _Python/_python_venvs/tensorflow_mac311
# Python version 3.11 for compatibility
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite