import os
import cv2
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from multiprocessing import Pool, cpu_count

# --- CONFIG ---
FOLDER_PATH = '/home/mbouchou/test_v2/' 
TFLITE_PATH = 'finetuned_model_512.tflite'
SUBMISSION_FILE = 'submissions/finetuned/rgb8_50.csv'
THRESHOLDS = [0.8] 
MIN_PIXELS = 50  # INCREASED SAFETY: Ignore any ship smaller than 80 pixels

# --- WORKER FUNCTION ---
def process_chunk(file_chunk):
    os.environ["SM_FRAMEWORK"] = "tf.keras"
    import tensorflow.keras as keras
    import tensorflow.keras.utils
    if not hasattr(keras.utils, 'generic_utils'):
        keras.utils.generic_utils = keras.utils
    # 1. Imports inside worker (Crucial for Parallel Stability)
    import tensorflow as tf
    import cv2
    import numpy as np
    from segmentation_models import get_preprocessing # USE THE LIBRARY
    
    # 2. Setup Model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h, w = input_details[0]['shape'][1], input_details[0]['shape'][2] # 512x512
    
    # 3. Setup Preprocessing (Exact match to training)
    # This handles the BGR/RGB and normalization automatically
    preprocess_input = get_preprocessing('resnet34')

    chunk_results = []
    
    for img_name in file_chunk:
        path = os.path.join(FOLDER_PATH, img_name)
        img = cv2.imread(path)
        if img is None:
            chunk_results.append({'ImageId': img_name, 'EncodedPixels': np.nan})
            continue

        # Resize to 512x512
        img_resized = cv2.resize(img, (w, h))
        
        # CRITICAL FIX: Keep it BGR (Standard OpenCV)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Convert to float32
        img_float = img_rgb.astype(np.float32)
        
        # Apply Library Preprocessing
        img_input = preprocess_input(img_float)
        img_input = np.expand_dims(img_input, axis=0)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        pred_map = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]

        # Post-Process
        # Resize mask back to 768x768
        pred_full = cv2.resize(pred_map, (768, 768), interpolation=cv2.INTER_LINEAR)
        binary_mask = (pred_full > THRESHOLDS[0]).astype(np.uint8)
        
        # EMPTY FILTER 1: Total Sum
        if binary_mask.sum() < MIN_PIXELS:
            chunk_results.append({'ImageId': img_name, 'EncodedPixels': np.nan})
            continue

        # Separate Components
        num_labels, labels_im = cv2.connectedComponents(binary_mask)
        
        if num_labels <= 1:
            chunk_results.append({'ImageId': img_name, 'EncodedPixels': np.nan})
        else:
            found_ship = False
            for label_id in range(1, num_labels):
                ship_mask = (labels_im == label_id).astype(np.uint8)
                
                # EMPTY FILTER 2: Individual Ship Size
                if ship_mask.sum() < MIN_PIXELS:
                    continue

                # RLE Encode
                pixels = ship_mask.T.flatten()
                pixels = np.concatenate([[0], pixels, [0]])
                runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
                runs[1::2] -= runs[::2]
                rle = " ".join(str(x) for x in runs)
                
                chunk_results.append({'ImageId': img_name, 'EncodedPixels': rle})
                found_ship = True
            
            if not found_ship:
                chunk_results.append({'ImageId': img_name, 'EncodedPixels': np.nan})
                
    return chunk_results

# --- MAIN CONTROLLER ---
def run_parallel_inference():
    print(f"🚀 Starting Corrected Inference (BGR Mode + Filter)...")
    
    files = sorted([f for f in os.listdir(FOLDER_PATH) if f.lower().endswith('.jpg')])
    total_files = len(files)
    
    num_cores = cpu_count()
    workers = max(1, num_cores - 1)
    print(f"Using {workers} Workers on {total_files} images.")
    
    # Split Data
    chunks = np.array_split(files, workers)
    
    start_time = time.time()
    
    # Run
    with Pool(processes=workers) as pool:
        results_list = pool.map(process_chunk, chunks)
            
    # Merge
    print("Merging results...")
    flat_results = [item for sublist in results_list for item in sublist]
    
    total_time = time.time() - start_time
    print(f"✅ FINISHED! Total Time: {total_time/60:.1f} minutes")
    
    # Save
    print(f"Saving to {SUBMISSION_FILE}...")
    output_dir = os.path.dirname(SUBMISSION_FILE)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
        
    pd.DataFrame(flat_results).to_csv(SUBMISSION_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    run_parallel_inference()