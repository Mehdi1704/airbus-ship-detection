import os
import cv2
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from skimage.measure import label, regionprops

# --- CONFIG ---
FOLDER_PATH = '/Users/mbouchou/Downloads/airbus-ship-detection/test_v2/'
TFLITE_PATH = 'finetuned_model_512.tflite' # The file we just created
SUBMISSION_FILE = 'submissions/finetuned/submission_final_04.csv'
THRESHOLDS = [0.4] # Stick to one for speed, or add more if needed

def rle_encode(mask):
    if mask.sum() == 0: return ""
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def run_fast_inference():
    # 1. Load TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape'] # e.g., [1, 256, 256, 3]
    print(input_shape)
    h, w = input_shape[1], input_shape[2]
    
    files = sorted([f for f in os.listdir(FOLDER_PATH) if f.endswith('.jpg')])
    results = {'ImageId': [], 'EncodedPixels': []}
    
    print(f"Processing {len(files)} images with TFLite...")
    start_time = time.time()

    for i, img_name in enumerate(files):
        # Progress check every 100 images
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            fps = i / elapsed
            eta = (len(files) - i) / fps / 60
            print(f"{i}/{len(files)} | {fps:.2f} FPS | ETA: {eta:.1f} min")

        # --- PREPROCESS ---
        path = os.path.join(FOLDER_PATH, img_name)
        img = cv2.imread(path)
        if img is None: continue
        
        # Resize to model input size (e.g. 256x256)
        img_resized = cv2.resize(img, (w, h)).astype(np.float32)
        # img_resized = img_resized / 255.0 # UNCOMMENT IF YOUR MODEL EXPECTS 0-1
        
        input_data = np.expand_dims(img_resized, axis=0)

        # --- INFERENCE ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        # --- POSTPROCESS ---
        # Resize prob map back to 768x768
        pred = pred[0, :, :, 0] # Remove batch and channel dims
        pred_full = cv2.resize(pred, (768, 768), interpolation=cv2.INTER_LINEAR)
        
        for thresh in THRESHOLDS:
            mask = (pred_full > thresh).astype(np.uint8)
            
            if mask.sum() == 0:
                results['ImageId'].append(img_name)
                results['EncodedPixels'].append("")
                continue

            # Labeling and RLE
            labeled_mask = label(mask)
            regions = regionprops(labeled_mask)
            
            if not regions:
                results['ImageId'].append(img_name)
                results['EncodedPixels'].append("")
                continue

            for region in regions:
                r_mask = (labeled_mask == region.label).astype(np.uint8)
                results['ImageId'].append(img_name)
                results['EncodedPixels'].append(rle_encode(r_mask))

    # Save
    pd.DataFrame(results).to_csv(SUBMISSION_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    run_fast_inference()