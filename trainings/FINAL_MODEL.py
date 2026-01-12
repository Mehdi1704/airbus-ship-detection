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
from segmentation_models import Unet, get_preprocessing

# --- CONFIGURATION ---
BATCH_SIZE = 64                 # Keep this high for V100
NB_EPOCHS = 15                  
CACHE_IMG_DIR = "/home/mbouchou/airbus-ship-detection-cache/images_256"
CACHE_MSK_DIR = "/home/mbouchou/airbus-ship-detection-cache/masks_256"
# Define preprocessing for ResNet34
PREPROCESS_INPUT = get_preprocessing('resnet34')

# --- FAST DATA LOADER (The Magic Part) ---

def load_npy_py_function(image_id_bytes, img_dir_bytes, msk_dir_bytes):
    """
    Python function to load .npy files, scale them, and apply preprocessing.
    This runs inside the tf.data pipeline.
    """
    # Decode bytes to strings
    img_id = image_id_bytes.numpy().decode('utf-8')
    img_dir = img_dir_bytes.numpy().decode('utf-8')
    msk_dir = msk_dir_bytes.numpy().decode('utf-8')

    img_path = os.path.join(img_dir, img_id + ".npy")
    msk_path = os.path.join(msk_dir, img_id + ".npy")

    # Load and scale
    img = np.load(img_path).astype(np.float32) * 255.0
    msk = np.load(msk_path).astype(np.float32)

    # Apply ResNet preprocessing
    img = PREPROCESS_INPUT(img)
    
    return img, msk

def tf_load_wrapper(image_id, img_dir, msk_dir):
    """
    TensorFlow wrapper that calls the python loading function.
    """
    img, msk = tf.py_function(
        func=load_npy_py_function,
        inp=[image_id, img_dir, msk_dir],
        Tout=[tf.float32, tf.float32]
    )
    # IMPORTANT: You must set the shape explicitly after py_function
    img.set_shape([256, 256, 3])
    msk.set_shape([256, 256, 1])
    return img, msk

def augment(img, msk):
    """
    Fast TensorFlow-native augmentation.
    """
    # Random Flip Left/Right
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        msk = tf.image.flip_left_right(msk)
    
    # Random Flip Up/Down
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        msk = tf.image.flip_up_down(msk)
        
    return img, msk

def create_dataset(df, img_dir, msk_dir, batch_size, is_training=False):
    # 1. Create dataset from Image IDs
    image_ids = df["ImageId"].drop_duplicates().values
    dataset = tf.data.Dataset.from_tensor_slices(image_ids)

    # 2. Map the loading function
    dataset = dataset.map(
        lambda img_id: tf_load_wrapper(img_id, img_dir, msk_dir),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --- THE MAGIC FIX: CACHE IN RAM ---
    # This stores the loaded data in memory after the first read.
    # Place this BEFORE shuffling and augmentation!
    dataset = dataset.cache() 
    # -----------------------------------

    # 3. Shuffle (only for training)
    if is_training:
        dataset = dataset.shuffle(buffer_size=2000)

    # 4. Augment (only for training)
    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # 5. Batch and Prefetch
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
    # Load DataFrame
    df = pd.read_csv("/home/mbouchou/airbus-ship-detection/masks_subset.csv")
    img_ids = df["ImageId"].drop_duplicates().values
    train_ids, val_ids = train_test_split(img_ids, test_size=0.2, random_state=42)

    train_df = df[df["ImageId"].isin(train_ids)]
    val_df   = df[df["ImageId"].isin(val_ids)]

    print("Creating fast tf.data pipelines...")
    train_ds, train_len = create_dataset(train_df, CACHE_IMG_DIR, CACHE_MSK_DIR, BATCH_SIZE, is_training=True)
    val_ds, val_len = create_dataset(val_df, CACHE_IMG_DIR, CACHE_MSK_DIR, BATCH_SIZE, is_training=False)

    # Calculate steps (needed because tf.data doesn't always know exact length)
    train_steps = int(np.ceil(train_len / BATCH_SIZE))
    val_steps = int(np.ceil(val_len / BATCH_SIZE))

    # Model
    seg_model = Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')
    
    seg_model.compile(
        optimizer=Adam(1e-3),
        loss=combo_loss,
        metrics=[dice_coef]
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        "seg_model_best_fast.keras",
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=3, verbose=1, mode='max', min_delta=1e-4, cooldown=2, min_lr=1e-6)
    early = EarlyStopping(monitor="val_dice_coef", mode="max", patience=15)

    print(f"Starting training on {train_len} images with Batch Size {BATCH_SIZE}...")
    
    history = seg_model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        epochs=NB_EPOCHS,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[checkpoint, reduceLROnPlat, early]
    )

    seg_model.save('seg_model_fast.keras')