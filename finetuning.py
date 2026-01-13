import os
# --- FRAMEWORK SETUP ---
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras import utils
import tensorflow.keras as keras
if not hasattr(keras.utils, 'generic_utils'):
    keras.utils.generic_utils = keras.utils

import tensorflow as tf
from tensorflow.keras import mixed_precision

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from segmentation_models import Unet, get_preprocessing

# --- CONFIGURATION ---
NEW_DIM = 512            # Target Resolution (Try 768 if you dare!)
BATCH_SIZE = 16          # 16-24 for 512px on V100
NB_EPOCHS = 15
LR = 1e-4

IMG_SOURCE_DIR = "/home/mbouchou/images"  # Path to original JPEGs
CSV_PATH = "/home/mbouchou/airbus-ship-detection/masks_subset.csv"

# Preprocessing
PREPROCESS_INPUT = get_preprocessing('resnet34')

# --- HELPER: FAST RLE DECODING ---
def rle_decode(mask_rle, shape=(768, 768)):
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    return img.reshape(shape).T # Airbus images are transposed

# --- DATA LOADING FUNCTION (The Heavy Lifter) ---
def load_and_process_py(image_id_bytes, rle_bytes):
    """
    Decodes JPEG and RLE, Resizes to NEW_DIM, Applies Preprocessing
    """
    img_id = image_id_bytes.numpy().decode('utf-8')
    # Combine all RLEs for this image (passed as a joined string)
    rle_full = rle_bytes.numpy().decode('utf-8')

    # 1. Load Image
    img_path = os.path.join(IMG_SOURCE_DIR, img_id)
    # Use OpenCV for speed, load as BGR then RGB
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # 2. Process Mask
    # We have to decode RLE manually. 
    # NOTE: This part is CPU intensive. If RLE is empty, return zeros.
    if rle_full == "nan" or rle_full == "":
        mask = np.zeros((768, 768, 1), dtype=np.float32)
    else:
        # Split joined RLEs if multiple ships
        # (We passed them joined by '|')
        rles = rle_full.split('|')
        mask = np.zeros((768, 768), dtype=np.uint8)
        for r in rles:
            mask += rle_decode(r, shape=(768, 768))
        
        # Clip to 0-1 (overlaps) and expand dims
        mask = np.clip(mask, 0, 1)[..., None].astype(np.float32)

    # 3. Resize BOTH to Target Dimension (e.g. 512x512)
    img = tf.image.resize(img, [NEW_DIM, NEW_DIM])
    mask = tf.image.resize(mask, [NEW_DIM, NEW_DIM], method='nearest')

    # 4. Preprocess Image (ResNet34)
    # Cast to float32 for preprocessing
    img = tf.cast(img, tf.float32)
    img = PREPROCESS_INPUT(img)
    
    return img, mask

def tf_load_wrapper(image_id, rle_str):
    img, msk = tf.py_function(
        func=load_and_process_py,
        inp=[image_id, rle_str],
        Tout=[tf.float32, tf.float32]
    )
    img.set_shape([NEW_DIM, NEW_DIM, 3])
    msk.set_shape([NEW_DIM, NEW_DIM, 1])
    return img, msk

def augment(img, msk):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        msk = tf.image.flip_left_right(msk)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        msk = tf.image.flip_up_down(msk)
    return img, msk

def create_dataset(df, batch_size, is_training=False):
    # Group RLEs by ImageId so we can pass them as a single string
    # Join multiple masks with a delimiter '|'
    grouped = df.groupby('ImageId')['EncodedPixels'].apply(
        lambda x: '|'.join([str(v) for v in x if pd.notna(v)])
    ).reset_index()
    
    # Fill empty with specific string for our decoder to recognize
    grouped['EncodedPixels'].replace('', 'nan', inplace=True)
    
    image_ids = grouped["ImageId"].values
    rle_strs = grouped["EncodedPixels"].values

    dataset = tf.data.Dataset.from_tensor_slices((image_ids, rle_strs))

    # Parallel loading is critical here since we are doing CPU work!
    dataset = dataset.map(
        tf_load_wrapper,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # NO CACHE (Direct from Disk)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, len(image_ids)

# --- LOSS FUNCTIONS ---
def dice_loss(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def combo_loss(y_true, y_pred, bce_weight=0.5, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce, axis=list(range(1, len(bce.shape))))
    dl = dice_loss(y_true, y_pred, smooth=smooth)
    return bce_weight * bce + (1.0 - bce_weight) * dl

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

# --- EXECUTION ---
if __name__ == "__main__":
    print(f"Initializing Direct-Read Training for {NEW_DIM}x{NEW_DIM}...")

    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    
    # 2. Filter (Ship Heavy)
    df['has_ship'] = df['EncodedPixels'].notna()
    ships = df[df['has_ship'] == True]
    empty = df[df['has_ship'] == False].sample(frac=0.10, random_state=42)
    df_balanced = pd.concat([ships, empty]).sample(frac=1).reset_index(drop=True)
    
    print(f"Dataset Balanced: {len(df_balanced)} images (90% Ships / 10% Empty)")

    # 3. Split
    unique_ids = df_balanced["ImageId"].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.1, random_state=42)

    train_df = df_balanced[df_balanced["ImageId"].isin(train_ids)]
    val_df   = df_balanced[df_balanced["ImageId"].isin(val_ids)]

    # 4. Create Pipelines
    # Note: We pass the DF directly now, not directories
    train_ds, train_len = create_dataset(train_df, BATCH_SIZE, is_training=True)
    val_ds, val_len = create_dataset(val_df, BATCH_SIZE, is_training=False)

    train_steps = int(np.ceil(train_len / BATCH_SIZE))
    val_steps = int(np.ceil(val_len / BATCH_SIZE))

    # 5. Model Setup
    print("Loading 256x256 Model Weights...")
    old_model = load_model('final_model.keras', custom_objects={'combo_loss': combo_loss, 'dice_coef': dice_coef}, compile=False)
    
    print(f"Building {NEW_DIM}x{NEW_DIM} Architecture...")
    seg_model = Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid', input_shape=(NEW_DIM, NEW_DIM, 3))
    
    print("💉 Transplanting Weights...")
    seg_model.set_weights(old_model.get_weights())
    
    seg_model.compile(
        optimizer=Adam(LR),
        loss=combo_loss,
        metrics=[dice_coef]
    )

    # 6. Train
    checkpoint = ModelCheckpoint(
        f"seg_model_{NEW_DIM}_finetuned.keras",
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=3, verbose=1, mode='max', min_lr=1e-6)
    early = EarlyStopping(monitor="val_dice_coef", mode="max", patience=8)

    print(f"Starting Fine-Tuning (Direct Read Mode)...")
    history = seg_model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        epochs=NB_EPOCHS,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[checkpoint, reduceLROnPlat, early]
    )