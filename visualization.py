import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import tensorflow as tf
from keras.models import load_model
from skimage.measure import label, regionprops

FOLDER_PATH ='/Users/mbouchou/Downloads/airbus-ship-detection/test_v2/'
THRESHOLD = 0.3
SUBMISSION_FILE = 'submission5.csv'
MODEL_PATH = '/Users/mbouchou/Desktop/airbus-ship-detection/best_unet_model3.keras'


def preprocess_input(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        return None  # signal "skip this file"

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def rle_encode(mask: np.ndarray) -> str:
    """
    Encode a binary mask into RLE, matching this decode:
        return img.reshape(shape).T

    mask: 2D array (H, W), values {0,1} or {False,True}
    returns: RLE string (start length start length ...)
    """
    if mask is None:
        return ""

    # Ensure 2D binary
    mask = (mask > 0).astype(np.uint8)

    # IMPORTANT: transpose to match your rle_decode (.T)
    pixels = mask.T.flatten()

    # Add sentinels at both ends
    pixels = np.concatenate([[0], pixels, [0]])

    # Find run starts/ends
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)

def make_submission(folder_path, model, threshold=THRESHOLD, batch_size=1):
    list_of_images = sorted(os.listdir(folder_path))
    image_id = []
    encoded_pixels = []

    batch_names = []
    batch_imgs = []
    
    # Timing counters
    time_preprocess = 0
    time_predict = 0
    time_postprocess = 0
    time_rle = 0

    def flush_batch():
        nonlocal batch_names, batch_imgs, image_id, encoded_pixels
        nonlocal time_preprocess, time_predict, time_postprocess, time_rle
        
        if not batch_imgs:
            return

        # PREDICTION
        t0 = time.time()
        X = np.concatenate(batch_imgs, axis=0)
        preds = model(X, training=False).numpy()
        time_predict += time.time() - t0

        # POSTPROCESSING
        t0 = time.time()
        for img_name, pred in zip(batch_names, preds):
            pred2d = np.squeeze(pred)
            if pred2d.ndim == 3:
                pred2d = pred2d[..., 0]

            # resize probabilities explicitly
            pred2d = cv2.resize(pred2d.astype(np.float32), (768, 768), interpolation=cv2.INTER_LINEAR)
            mask = (pred2d > threshold).astype(np.uint8)

            if mask.sum() == 0:
                image_id.append(img_name)
                encoded_pixels.append("")
                continue

            labeled_mask = label(mask)
            regions = list(regionprops(labeled_mask))

            if len(regions) == 0:
                image_id.append(img_name)
                encoded_pixels.append("")
                continue

            # RLE ENCODING
            t_rle_start = time.time()
            for region in regions:
                single_ship_mask = (labeled_mask == region.label).astype(np.uint8)
                image_id.append(img_name)
                encoded_pixels.append(rle_encode(single_ship_mask))
            time_rle += time.time() - t_rle_start
        
        time_postprocess += time.time() - t0

        batch_names, batch_imgs = [], []

    # MAIN LOOP
    t_loop_start = time.time()
    for img_name in list_of_images:
        t0 = time.time()
        x = preprocess_input(os.path.join(folder_path, img_name))
        time_preprocess += time.time() - t0
        
        if x is None:
            continue

        batch_names.append(img_name)
        batch_imgs.append(x)

        if len(batch_imgs) >= batch_size:
            flush_batch()

    flush_batch()
    t_loop_end = time.time()

    # PRINT TIMING REPORT
    print("\n" + "="*60)
    print("TIMING REPORT")
    print("="*60)
    print(f"Image Preprocessing (load+resize): {time_preprocess:.2f}s")
    print(f"Model Prediction:                   {time_predict:.2f}s")
    print(f"Postprocessing (resize+threshold): {time_postprocess - time_rle:.2f}s")
    print(f"RLE Encoding:                       {time_rle:.2f}s")
    print(f"Total Loop Time:                    {t_loop_end - t_loop_start:.2f}s")
    print("="*60)

    return pd.DataFrame({"ImageId": image_id, "EncodedPixels": encoded_pixels})


model = load_model(
    MODEL_PATH, compile=False)
submission = make_submission(FOLDER_PATH, model, threshold=THRESHOLD)    
submission.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission file created: {SUBMISSION_FILE}")