import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# --- 1. SETUP ENV & MONKEY PATCH ---
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow.keras as keras
import tensorflow.keras.utils
if not hasattr(keras.utils, 'generic_utils'):
    keras.utils.generic_utils = keras.utils

from segmentation_models import get_preprocessing

# --- 2. CONFIG ---
FOLDER_PATH = '/home/mbouchou/test_v2/'
MODEL_PATH = 'final_model.tflite'
OUTPUT_DIR = 'visual_debug_old_50'
NUM_IMAGES = 50
IMAGES_PER_SHEET = 10

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. LOAD MODEL ---
print(f"Loading model: {MODEL_PATH}...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get model input shape automatically (e.g., 256 or 512)
h_model, w_model = input_details[0]['shape'][1], input_details[0]['shape'][2]
print(f"Model expects input size: {h_model}x{w_model}")

# Preprocessing (ResNet34 expects RGB)
preprocess_input = get_preprocessing('resnet34')

# --- 4. GET IMAGES ---
all_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(('.jpg', '.png'))]
if len(all_files) < NUM_IMAGES:
    selected_files = all_files
else:
    selected_files = random.sample(all_files, NUM_IMAGES)

print(f"Selected {len(selected_files)} images for visualization.")

# --- 5. RUN & PLOT ---
# We process in batches of 10 to create "Summary Sheets"
for sheet_idx in range(0, len(selected_files), IMAGES_PER_SHEET):
    batch_files = selected_files[sheet_idx : sheet_idx + IMAGES_PER_SHEET]
    
    # Create a figure with 10 rows, 2 columns
    fig, axes = plt.subplots(len(batch_files), 2, figsize=(10, 4 * len(batch_files)))
    fig.suptitle(f'Batch {sheet_idx//IMAGES_PER_SHEET + 1} (Images {sheet_idx+1}-{sheet_idx+len(batch_files)})', fontsize=16)
    
    for row, img_name in enumerate(batch_files):
        # Load Image
        path = os.path.join(FOLDER_PATH, img_name)
        img = cv2.imread(path)
        
        # Resize to model target
        img_resized = cv2.resize(img, (w_model, h_model))
        
        # --- CRITICAL: COLOR FIX FOR INFERENCE ---
        # OpenCV loads as BGR. We convert to RGB for the model.
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        img_float = img_rgb.astype(np.float32)
        img_input = preprocess_input(img_float)
        img_input = np.expand_dims(img_input, axis=0)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        pred_mask = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]

        # --- PLOTTING ---
        # Column 1: Original Image
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"Original: {img_name}", fontsize=10)
        axes[row, 0].axis('off')

        # Column 2: Prediction
        # Using 'jet' map: Blue is low confidence, Red is high confidence
        axes[row, 1].imshow(pred_mask, cmap='jet', vmin=0, vmax=1)
        axes[row, 1].set_title("Prediction (Blue=0, Red=1)", fontsize=10)
        axes[row, 1].axis('off')

    # Save the sheet
    output_path = os.path.join(OUTPUT_DIR, f"debug_sheet_old_{sheet_idx//IMAGES_PER_SHEET + 1}.jpg")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95) # Make room for title
    plt.savefig(output_path)
    plt.close(fig) # Free memory
    
    print(f"Saved {output_path}")

print("✅ Visualization Complete!")