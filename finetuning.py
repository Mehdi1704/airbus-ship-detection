import os
# --- FRAMEWORK SETUP ---
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras import utils
import tensorflow.keras as keras
if not hasattr(keras.utils, 'generic_utils'):
    keras.utils.generic_utils = keras.utils

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from segmentation_models import Unet, get_preprocessing

# --- CONFIGURATION ---
# CRITICAL: Set this to match the cache you just generated (512 or 768)
NEW_DIM = 512  
BATCH_SIZE = 8   # Lower batch size to prevent OOM (8 for 512px, 4 for 768px)
NB_EPOCHS = 15   
LR = 1e-4        # Low Learning Rate for Fine-Tuning

# Update paths to your NEW High-Res Cache
CACHE_IMG_DIR = f"/home/mbouchou/airbus-ship-detection-cache/images_{NEW_DIM}"
CACHE_MSK_DIR = f"/home/mbouchou/airbus-ship-detection-cache/masks_{NEW_DIM}"

# Define preprocessing for ResNet34
PREPROCESS_INPUT = get_preprocessing('resnet34')

# --- FAST DATA LOADER ---
def load_npy_py_function(image_id_bytes, img_dir_bytes, msk_dir_bytes):
    img_id = image_id_bytes.numpy().decode('utf-8')
    img_dir = img_dir_bytes.numpy().decode('utf-8')
    msk_dir = msk_dir_bytes.numpy().decode('utf-8')

    img_path = os.path.join(img_dir, img_id + ".npy")
    msk_path = os.path.join(msk_dir, img_id + ".npy")

    # Load and scale (Assuming cache is 0-1 float16, we convert to 0-255 for ResNet)
    img = np.load(img_path).astype(np.float32) * 255.0
    msk = np.load(msk_path).astype(np.float32)

    img = PREPROCESS_INPUT(img)
    return img, msk

def tf_load_wrapper(image_id, img_dir, msk_dir):
    img, msk = tf.py_function(
        func=load_npy_py_function,
        inp=[image_id, img_dir, msk_dir],
        Tout=[tf.float32, tf.float32]
    )
    # IMPORTANT: Set shape to NEW_DIM
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

def create_dataset(df, img_dir, msk_dir, batch_size, is_training=False):
    image_ids = df["ImageId"].drop_duplicates().values
    dataset = tf.data.Dataset.from_tensor_slices(image_ids)

    dataset = dataset.map(
        lambda img_id: tf_load_wrapper(img_id, img_dir, msk_dir),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # SAFETY: Disabled .cache() for High-Res to prevent RAM Crash.
    # If you have >64GB RAM, you can uncomment it.
    # dataset = dataset.cache() 

    if is_training:
        dataset = dataset.shuffle(buffer_size=500)
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
    # 1. Load Data
    print("Loading Dataframe...")
    df = pd.read_csv("/home/mbouchou/airbus-ship-detection/masks_subset.csv")
    
    # --- STRATEGY: SHIP-HEAVY DIET ---
    # We keep 100% of ships and only 10% of empty images
    print("Filtering dataset for Fine-Tuning...")
    df['has_ship'] = df['EncodedPixels'].notna()
    ships = df[df['has_ship'] == True]
    empty = df[df['has_ship'] == False].sample(frac=0.10, random_state=42)
    
    # Recombine and shuffle
    df_balanced = pd.concat([ships, empty]).sample(frac=1).reset_index(drop=True)
    print(f"Dataset Balanced: {len(df_balanced)} images (Mostly Ships)")

    # 2. Split
    img_ids = df_balanced["ImageId"].drop_duplicates().values
    train_ids, val_ids = train_test_split(img_ids, test_size=0.1, random_state=42)

    train_df = df_balanced[df_balanced["ImageId"].isin(train_ids)]
    val_df   = df_balanced[df_balanced["ImageId"].isin(val_ids)]

    # 3. Pipelines
    print(f"Creating tf.data pipelines reading from {CACHE_IMG_DIR}...")
    train_ds, train_len = create_dataset(train_df, CACHE_IMG_DIR, CACHE_MSK_DIR, BATCH_SIZE, is_training=True)
    val_ds, val_len = create_dataset(val_df, CACHE_IMG_DIR, CACHE_MSK_DIR, BATCH_SIZE, is_training=False)

    train_steps = int(np.ceil(train_len / BATCH_SIZE))
    val_steps = int(np.ceil(val_len / BATCH_SIZE))

    # --- 4. BRAIN TRANSPLANT ---
    print("Loading OLD model weights (256x256)...")
    # We load the weights from your best previous run
    old_model = load_model('seg_model_best_fast.keras', custom_objects={'combo_loss': combo_loss, 'dice_coef': dice_coef}, compile=False)
    
    print(f"Building NEW model ({NEW_DIM}x{NEW_DIM})...")
    # We build a fresh model container with the larger input shape
    seg_model = Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid', input_shape=(NEW_DIM, NEW_DIM, 3))
    
    print("Transplanting weights...")
    # Transfer the intelligence
    seg_model.set_weights(old_model.get_weights())
    
    # Compile with LOW Learning Rate
    seg_model.compile(
        optimizer=Adam(LR),
        loss=combo_loss,
        metrics=[dice_coef]
    )

    # 5. Train
    checkpoint = ModelCheckpoint(
        f"seg_model_{NEW_DIM}_finetuned.keras", # Saves to a new file
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=3, verbose=1, mode='max', min_lr=1e-6)
    early = EarlyStopping(monitor="val_dice_coef", mode="max", patience=8)

    print(f"Starting High-Res Fine-Tuning...")
    
    history = seg_model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        epochs=NB_EPOCHS,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[checkpoint, reduceLROnPlat, early]
    )
    
    print("Done! Convert this new model to TFLite and submit.")