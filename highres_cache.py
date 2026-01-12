import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------- CONFIG ---------
# 1. CHANGE THIS TO 512 or 768
IMAGE_SIZE = (512, 512) 

IMG_DIR = "/home/mbouchou/images"
CSV_PATH = "/home/mbouchou/airbus-ship-detection/masks_subset.csv"

# 2. SAVE TO A NEW FOLDER
OUT_IMG_DIR = f"/home/mbouchou/airbus-ship-detection-cache/images_{IMAGE_SIZE[0]}"
OUT_MSK_DIR = f"/home/mbouchou/airbus-ship-detection-cache/masks_{IMAGE_SIZE[0]}"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MSK_DIR, exist_ok=True)

def rle_decode(mask_rle, shape=(768, 768)):
    if not isinstance(mask_rle, str):
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

print(f"Reading CSV from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# --- SMART FILTERING ---
# We want to train mostly on ships now to fix recall.
# 1. Get IDs of images with ships
df['has_ship'] = df['EncodedPixels'].notna()
ship_ids = df[df['has_ship']]['ImageId'].unique()

# 2. Get IDs of empty images (Sample only 5-10% of them)
empty_ids = df[~df['has_ship']]['ImageId'].unique()
# Randomly sample 10% of empty images to keep the model honest
np.random.seed(42)
empty_ids_subset = np.random.choice(empty_ids, size=int(len(empty_ids) * 0.10), replace=False)

# 3. Combine
ids_to_cache = np.concatenate([ship_ids, empty_ids_subset])
print(f"Caching {len(ids_to_cache)} images ({len(ship_ids)} ships, {len(empty_ids_subset)} empty).")
print(f"Target Resolution: {IMAGE_SIZE}")

masks_by_image = df.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()

for image_id in tqdm(ids_to_cache, desc="Generating High-Res Cache"):
    img_path = os.path.join(IMG_DIR, image_id)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: continue

    # RESIZE to 512 or 768
    img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img = (img.astype(np.float32) / 255.0).astype(np.float16)
    np.save(os.path.join(OUT_IMG_DIR, image_id + ".npy"), img)

    # Process MASK
    mask = np.zeros((768, 768), dtype=np.uint8)
    for rle in masks_by_image.get(image_id, []):
        if isinstance(rle, str):
            mask |= rle_decode(rle, shape=(768, 768)).astype(np.uint8)

    # Resize mask (Be careful with interpolation!)
    # INTER_NEAREST is correct for masks, but at 512x512 it's much safer than 256.
    mask = cv2.resize(mask, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask[..., None].astype(np.uint8)
    np.save(os.path.join(OUT_MSK_DIR, image_id + ".npy"), mask)

print("✅ High-Res Cache Generation Complete.")