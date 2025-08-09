import numpy as np
import tensorflow as tf

# === 1. Create some training data (sine wave) ===
X = np.linspace(-np.pi, np.pi, 200, dtype=np.float32)
y = np.sin(X)

# === 2. Define a very small model ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# === 3. Compile & train briefly ===
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# === 4. Convert to TensorFlow Lite ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# === 5. Save to file ===
with open("hello_world.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved hello_world.tflite, size:", len(tflite_model), "bytes")