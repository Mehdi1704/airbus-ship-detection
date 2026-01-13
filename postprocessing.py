import pandas as pd
import numpy as np
import os
import gc

# --- CONFIGURATION ---
INPUT_CSV = 'submissions/finetuned/submission_v2_corrected_09.csv'
OUTPUT_DIR = 'submissions/finetuned_09' # Folder to save variants
IMG_SHAPE = (768, 768)

# Define all the pixel thresholds you want to generate files for
PIXEL_THRESHOLDS = [200, 400, 600]

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---
def rle_decode(mask_rle):
    if pd.isna(mask_rle) or mask_rle == '':
        return None
    try:
        s = mask_rle.split()
        starts = np.array([int(x) for x in s[0::2]])
        lengths = np.array([int(x) for x in s[1::2]])
        starts -= 1
        ends = starts + lengths
        mask = np.zeros(IMG_SHAPE[0] * IMG_SHAPE[1], dtype=bool)
        for lo, hi in zip(starts, ends):
            lo = max(0, lo)
            hi = min(mask.size, hi)
            if lo < hi:
                mask[lo:hi] = True
        return mask
    except:
        return None

def rle_encode(mask):
    pixels = mask.astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def process_group_for_threshold(group, min_pixels):
    """
    Process image group for a SPECIFIC pixel threshold.
    """
    valid_masks = []
    
    # 1. Decode & Initial Filter
    for rle in group['EncodedPixels']:
        # Optimization: We decode once, but we could cache decoded masks 
        # outside this loop if memory allows. For safety, we decode here.
        mask = rle_decode(rle)
        if mask is not None and mask.sum() >= min_pixels:
            valid_masks.append(mask)
            
    if not valid_masks:
        return []

    # 2. Instance Separation (Subtract Overlaps)
    final_rles = []
    occupied = np.zeros(IMG_SHAPE[0] * IMG_SHAPE[1], dtype=bool)
    
    for mask in valid_masks:
        # Subtract
        mask = mask & (~occupied)
        
        # Check size AGAIN after trimming overlap
        if mask.sum() >= min_pixels:
            occupied |= mask
            final_rles.append(rle_encode(mask))
            
    return final_rles

# --- MAIN EXECUTION ---
print(f"Reading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
all_images = df['ImageId'].unique()
grouped = df.groupby('ImageId')

print(f"Loaded {len(all_images)} images. Generating {len(PIXEL_THRESHOLDS)} variants...")

# Dictionary to hold lists of rows for each threshold
# structure: { 100: [row1, row2...], 200: [row1, row2...] }
results = {p: [] for p in PIXEL_THRESHOLDS}

count = 0
total_imgs = len(grouped)

# Process each image ONCE, but apply logic for ALL thresholds
for image_id, group in grouped:
    # 1. Decode all masks for this image first (Optimization)
    raw_masks = []
    for rle in group['EncodedPixels']:
        m = rle_decode(rle)
        if m is not None:
            raw_masks.append(m)
            
    # 2. For each threshold, apply logic
    for p in PIXEL_THRESHOLDS:
        # Filter raw masks by size 'p'
        candidates = [m for m in raw_masks if m.sum() >= p]
        
        final_rles = []
        if candidates:
            occupied = np.zeros(IMG_SHAPE[0] * IMG_SHAPE[1], dtype=bool)
            for mask in candidates:
                # Subtract overlaps
                clean_mask = mask & (~occupied)
                
                # Check trimmed size
                if clean_mask.sum() >= p:
                    occupied |= clean_mask
                    final_rles.append(rle_encode(clean_mask))
        
        # Add to results
        if not final_rles:
            results[p].append({'ImageId': image_id, 'EncodedPixels': np.nan})
        else:
            for rle in final_rles:
                results[p].append({'ImageId': image_id, 'EncodedPixels': rle})

    count += 1
    if count % 500 == 0:
        print(f"Processed {count}/{total_imgs} images...")

# --- SAVE ALL FILES ---
print("\nSaving files...")

for p in PIXEL_THRESHOLDS:
    rows = results[p]
    sub_df = pd.DataFrame(rows)
    
    # Restore missing IDs if any
    missing_ids = set(all_images) - set(sub_df['ImageId'].unique())
    if missing_ids:
        print(f"  [MinPixels={p}] Restoring {len(missing_ids)} empty IDs...")
        missing_rows = [{'ImageId': i, 'EncodedPixels': np.nan} for i in missing_ids]
        sub_df = pd.concat([sub_df, pd.DataFrame(missing_rows)], ignore_index=True)
        
    filename = f"{OUTPUT_DIR}/submission_04_clean_{p}.csv"
    sub_df.to_csv(filename, index=False)
    print(f"✅ Saved: {filename} (Total Rows: {len(sub_df)})")

print("\nDone! All variants generated.")