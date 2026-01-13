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
# We test 0.85 and 0.90. TTA reduces noise, so 0.85 becomes safer to use.
TEST_THRESHOLDS = [0.80, 0.85, 0.90] 
MIN_PIXELS = 50 

def process_chunk(file_chunk):
    # --- 1. WORKER SETUP (Monkey Patch Included) ---
    import os
    os.environ["SM_FRAMEWORK"] = "tf.keras"
    import tensorflow.keras as keras
    import tensorflow.keras.utils
    if not hasattr(keras.utils, 'generic_utils'):
        keras.utils.generic_utils = keras.utils

    import tensorflow as tf
    import cv2
    import numpy as np
    from segmentation_models import get_preprocessing

    # Load Model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    
    preprocess_input = get_preprocessing('resnet34')
    
    chunk_results = {t: [] for t in TEST_THRESHOLDS}
    
    for img_name in file_chunk:
        path = os.path.join(FOLDER_PATH, img_name)
        img = cv2.imread(path)
        if img is None:
            for t in TEST_THRESHOLDS:
                chunk_results[t].append({'ImageId': img_name, 'EncodedPixels': np.nan})
            continue

        # --- PREPARE INPUT ---
        img_resized = cv2.resize(img, (w, h))
        img_float = img_resized.astype(np.float32)
        # Preprocess first (mean subtraction)
        img_input = preprocess_input(img_float)
        
        # --- PASS 1: ORIGINAL ---
        batch_1 = np.expand_dims(img_input, axis=0)
        interpreter.set_tensor(input_details[0]['index'], batch_1)
        interpreter.invoke()
        # Shape: (512, 512)
        pred_1 = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]
        
        # --- PASS 2: FLIPPED (TTA) ---
        # Flip the INPUT image horizontally (axis 1)
        img_flipped = cv2.flip(img_input, 1) 
        batch_2 = np.expand_dims(img_flipped, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], batch_2)
        interpreter.invoke()
        pred_flipped = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]
        
        # Flip the OUTPUT prediction back so it matches the original
        pred_2 = cv2.flip(pred_flipped, 1)
        
        # --- AVERAGE ---
        # This is where the magic happens. 
        # Real ships stay high (0.9 + 0.9) / 2 = 0.9
        # Random noise drops (0.7 + 0.1) / 2 = 0.4 -> Eliminated by threshold!
        pred_avg = (pred_1 + pred_2) / 2.0

        # Resize to full resolution
        pred_full = cv2.resize(pred_avg, (768, 768), interpolation=cv2.INTER_LINEAR)
        
        # --- THRESHOLDING ---
        for thresh in TEST_THRESHOLDS:
            binary_mask = (pred_full > thresh).astype(np.uint8)
            
            # Global Filter
            if binary_mask.sum() < MIN_PIXELS:
                chunk_results[thresh].append({'ImageId': img_name, 'EncodedPixels': np.nan})
                continue

            num_labels, labels_im = cv2.connectedComponents(binary_mask)
            
            if num_labels <= 1:
                chunk_results[thresh].append({'ImageId': img_name, 'EncodedPixels': np.nan})
            else:
                found_ship = False
                for label_id in range(1, num_labels):
                    ship_mask = (labels_im == label_id).astype(np.uint8)
                    
                    # Individual Filter
                    if ship_mask.sum() < MIN_PIXELS: continue

                    pixels = ship_mask.T.flatten()
                    pixels = np.concatenate([[0], pixels, [0]])
                    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
                    runs[1::2] -= runs[::2]
                    rle = " ".join(str(x) for x in runs)
                    
                    chunk_results[thresh].append({'ImageId': img_name, 'EncodedPixels': rle})
                    found_ship = True
                
                if not found_ship:
                    chunk_results[thresh].append({'ImageId': img_name, 'EncodedPixels': np.nan})
                
    return chunk_results

def run_tta_inference():
    print(f"🚀 Running TTA (Test Time Augmentation) 2-Pass Inference...")
    print(f"Testing Thresholds: {TEST_THRESHOLDS}")
    
    files = sorted([f for f in os.listdir(FOLDER_PATH) if f.lower().endswith('.jpg')])
    
    num_cores = cpu_count()
    workers = max(1, num_cores - 1)
    print(f"Using {workers} Workers.")
    
    chunks = np.array_split(files, workers)
    
    start_time = time.time()
    
    with Pool(processes=workers) as pool:
        results_list = pool.map(process_chunk, chunks)
            
    print("Merging results...")
    
    final_data = {t: [] for t in TEST_THRESHOLDS}
    
    for worker_output in results_list:
        for t in TEST_THRESHOLDS:
            final_data[t].extend(worker_output[t])
            
    total_time = time.time() - start_time
    print(f"✅ FINISHED! Total Time: {total_time/60:.1f} minutes")

    for t in TEST_THRESHOLDS:
        filename = f"submission_tta_thresh_{t}.csv"
        print(f"Saving {filename}...")
        pd.DataFrame(final_data[t]).to_csv(filename, index=False)
        
    print("Done! Submit the 0.85 file first.")

if __name__ == "__main__":
    run_tta_inference()