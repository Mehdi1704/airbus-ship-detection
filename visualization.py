import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

from skimage.measure import label

folder_path='/home/mbouchou/images'
img_name = '0a0ada7e3.jpg'
# Load and preprocess the input image 
def preprocess_input(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        print("Error: Loaded image has invalid dimensions.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def rle_decode(mask_rle, shape=(768, 768)):
    if not isinstance(mask_rle, str):
        return np.zeros(shape)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

model = load_model(
    "/home/mbouchou/airbus-ship-detection/best_unet_model2.keras",   # or .h5
    custom_objects={"dice_coefficient": dice_coefficient}
)
img_path = os.path.join(folder_path, img_name)
img = np.squeeze(preprocess_input(img_path), axis=0)
mask = model.predict(preprocess_input(img_path), verbose=0)
mask = (mask > 0.3).astype(int)


one_sample_masks = img_name

# Set the figure size
plt.figure(figsize=(15, 10))  # Adjust the width and height as needed

# Row 1
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(np.squeeze(mask, axis=0))
plt.title('Model Output')

plt.subplot(2, 3, 3)
plt.imshow(label(np.squeeze(mask, axis=0)))
plt.title('Labeled Output')

# Row 2
plt.subplot(2, 3, 4)
plt.imshow(rle_decode(one_sample_masks.EncodedPixels.iloc[0]))
plt.title('Decoded Mask 1')

plt.subplot(2, 3, 5)
plt.imshow(rle_decode(one_sample_masks.EncodedPixels.iloc[1]))
plt.title('Decoded Mask 2')

plt.subplot(2, 3, 6)
plt.imshow(rle_decode(one_sample_masks.EncodedPixels.iloc[2]))
plt.title('Decoded Mask 3')

# Show the plot
plt.show()