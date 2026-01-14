import cv2
import numpy as np
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow.keras as keras
import tensorflow.keras.utils
if not hasattr(keras.utils, 'generic_utils'):
    keras.utils.generic_utils = keras.utils
import tensorflow as tf
from segmentation_models import get_preprocessing
import matplotlib.pyplot as plt

# Load TFLite
interpreter = tf.lite.Interpreter(model_path="final_model.tflite")
interpreter.allocate_tensors()
preprocess_input = get_preprocessing('resnet34')

# Load 1 image
img = cv2.imread("/home/mbouchou/test_v2/6f5f44ca1.jpg") # Pick a random one
img_resized = cv2.resize(img, (256, 256))
img_float = img_resized.astype(np.float32)
img_input = preprocess_input(img_float)
img_input = np.expand_dims(img_input, axis=0)

# Run
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_input)
interpreter.invoke()
pred = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0,:,:,0]

# Save
plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(pred, cmap='jet'); plt.title("Prediction")
plt.savefig("debug_output_old2.jpg")
print("Saved debug_output.jpg")