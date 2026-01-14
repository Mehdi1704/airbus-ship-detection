import tensorflow as tf
import keras
import numpy as np

# --- CONFIG ---
MODEL_PATH = '/Users/mbouchou/Downloads/final_model.keras'
TFLITE_PATH = 'final_model.tflite'
INPUT_SHAPE = (1, 256, 256, 3) # (Batch, Height, Width, Channels)

print(f"Loading Keras model from {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH, compile=False)

# --- THE FIX: WRAP IN A CONCRETE FUNCTION ---
# This wrapper explicitly defines the input signature and forces training=False.
# It effectively "freezes" the Batch Normalization layers before TFLite sees them.
class ModelWrapper(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32)])
    def serve(self, x):
        # We enforce training=False here to fix the "ReadVariableOp" error
        return self.model(x, training=False)

print("Creating concrete function...")
wrapper = ModelWrapper(model)
concrete_func = wrapper.serve.get_concrete_function()

# --- CONVERT ---
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# Optional: Enable default optimizations (usually safe and faster)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# --- SAVE ---
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Success! Saved to {TFLITE_PATH}")
print(f"Model Size: {len(tflite_model) / 1024 / 1024:.2f} MB")