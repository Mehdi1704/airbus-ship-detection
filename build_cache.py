import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------- CONFIG ---------
IMG_DIR = "/home/mbouchou/images"
CSV_PATH = "/home/mbouchou/airbus-ship-detection/masks_subset.csv"

OUT_IMG_DIR = "/home/mbouchou/airbus-ship-detection-cache/images_256"
OUT_MSK_DIR = "/home/mbouchou/airbus-ship-detection-cache/masks_256"

IMAGE_SIZE = (256, 256)  # (H, W)
# --------------------------

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
    return img.reshape(shape).T  # Airbus convention

df = pd.read_csv(CSV_PATH)

# One row per image for caching
image_ids = df["ImageId"].drop_duplicates().values

# Group all RLEs per image once
masks_by_image = df.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()

for image_id in tqdm(sorted(image_ids), desc="Caching npy"):
    img_path = os.path.join(IMG_DIR, image_id)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        # skip missing/corrupt files
        continue

    # Image -> 256x256, float16 to reduce disk + IO
    img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img = (img.astype(np.float32) / 255.0).astype(np.float16)
    np.save(os.path.join(OUT_IMG_DIR, image_id + ".npy"), img)

    # Mask -> decode at 768, merge ships, resize to 256, store uint8 (0/1)
    mask = np.zeros((768, 768), dtype=np.uint8)
    for rle in masks_by_image.get(image_id, []):
        if isinstance(rle, str):
            mask |= rle_decode(rle, shape=(768, 768)).astype(np.uint8)

    mask = cv2.resize(mask, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask[..., None].astype(np.uint8)  # (256,256,1)
    np.save(os.path.join(OUT_MSK_DIR, image_id + ".npy"), mask)

print("Done. Cached:")
print("Images:", OUT_IMG_DIR)
print("Masks :", OUT_MSK_DIR)