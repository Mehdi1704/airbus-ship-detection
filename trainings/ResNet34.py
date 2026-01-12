import os
# 1. Set the framework to use TensorFlow Keras explicitly
os.environ["SM_FRAMEWORK"] = "tf.keras"

# 2. Fix the missing generic_utils attribute (Monkey Patch)
import tensorflow.keras.utils
import tensorflow.keras as keras

# Create a dummy generic_utils module if it doesn't exist
# This tricks the library into thinking the old structure is still there
if not hasattr(keras.utils, 'generic_utils'):
    keras.utils.generic_utils = keras.utils

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from keras import models, layers
# Now it is safe to import segmentation_models
from segmentation_models import Unet, get_preprocessing
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from npy_generator import CachedNpyGenerator
from tensorflow.keras.optimizers import Adam

# Parameters
BATCH_SIZE = 64                 
NB_EPOCHS = 15                 
CACHE_IMG_DIR = "/home/mbouchou/airbus-ship-detection-cache/images_256"
CACHE_MSK_DIR = "/home/mbouchou/airbus-ship-detection-cache/masks_256"

# --- MODEL DEFINITION ---
# This one line replaces all the manual layer building you had before.
# It downloads ResNet34 weights trained on ImageNet.
seg_model = Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')
preprocess_input = get_preprocessing('resnet34')

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

# --- DATA LOADING ---
df = pd.read_csv("/home/mbouchou/airbus-ship-detection/masks_subset.csv")
img_ids = df["ImageId"].drop_duplicates().values
train_ids, val_ids = train_test_split(img_ids, test_size=0.2, random_state=42)

train_df = df[df["ImageId"].isin(train_ids)]
val_df   = df[df["ImageId"].isin(val_ids)]

train_gen = CachedNpyGenerator(
    train_df, 
    CACHE_IMG_DIR, 
    CACHE_MSK_DIR, 
    batch_size=BATCH_SIZE, 
    shuffle=True,  
    augment=True,
    preprocess=preprocess_input  # <--- PASS IT HERE
)

val_gen = CachedNpyGenerator(
    val_df,   
    CACHE_IMG_DIR, 
    CACHE_MSK_DIR, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    augment=False,
    preprocess=preprocess_input  # <--- PASS IT HERE
)

# --- COMPILE ---
seg_model.compile(
    optimizer=Adam(1e-3),  # 1e-3 is good for ResNet decoder
    loss=combo_loss,
    metrics=[dice_coef]
)

# --- CALLBACKS ---
checkpoint = ModelCheckpoint(
    "seg_model_best_new.keras",
    monitor="val_dice_coef",
    mode="max",
    save_best_only=True,
    verbose=1
)

reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_dice_coef',
    factor=0.5,
    patience=3,
    verbose=1,
    mode='max',
    min_delta=1e-4,
    cooldown=2,
    min_lr=1e-6
)

early = EarlyStopping(monitor="val_dice_coef", mode="max", patience=15)

callbacks_list = [checkpoint, early, reduceLROnPlat]

# --- TRAINING ---
# Using the generator length directly is the safest way
steps_per_epoch = len(train_gen)
validation_steps = len(val_gen)

print(f"Training on {steps_per_epoch} steps per epoch")
print(f"Validating on {validation_steps} steps")

history = seg_model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=NB_EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks_list
)

seg_model.save('seg_model_new.keras')