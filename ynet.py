from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from keras import models, layers
import keras.backend as K
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from npy_generator import CachedNpyGenerator


# Parameters
BATCH_SIZE = 4                 # Train batch size
EDGE_CROP = 16                 # While building the model
NB_EPOCHS = 5                  # Training epochs
GAUSSIAN_NOISE = 0.1           # To be used in a layer in the model
UPSAMPLE_MODE = 'SIMPLE'       # SIMPLE ==> UpSampling2D, else Conv2DTranspose
NET_SCALING = None             # Downsampling inside the network                        
IMG_SCALING = (1, 1)           # Downsampling in preprocessing
VALID_IMG_COUNT = 400          # Valid batch size
MAX_TRAIN_STEPS = 200          # Maximum number of steps_per_epoch in training
CACHE_IMG_DIR = "/home/mbouchou/airbus-ship-detection-cache/images_256"
CACHE_MSK_DIR = "/home/mbouchou/airbus-ship-detection-cache/masks_256"

# Conv2DTranspose upsampling
def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
# Upsampling without Conv2DTranspose
def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)

# Upsampling method choice
if UPSAMPLE_MODE=='DECONV':
    upsample=upsample_conv
else:
    upsample=upsample_simple

# Building the layers of UNET
input_img = layers.Input((256, 256, 3), name='RGB_Input')
pp_in_layer = input_img

# If NET_SCALING is defined then do the next step else continue ahead
if NET_SCALING is not None:
    pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

# To avoid overfitting and fastening the process of training
pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)                       # Useful to mitigate overfitting
pp_in_layer = layers.BatchNormalization()(pp_in_layer)                                # Allows using higher learning rate without causing problems with gradients


## Downsample (C-->C-->MP)

c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (pp_in_layer)
c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = layers.MaxPooling2D((2, 2)) (c1)

c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = layers.MaxPooling2D((2, 2)) (c2)

c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = layers.MaxPooling2D((2, 2)) (c3)

c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)


c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

## Upsample (U --> Concat --> C --> C)

u6 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = upsample(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)

if NET_SCALING is not None:
    d = layers.UpSampling2D(NET_SCALING)(d)

seg_model = models.Model(inputs=[input_img], outputs=[d])

seg_model.summary()

def dice_loss(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice  # (batch,)

def combo_loss(y_true, y_pred, bce_weight=0.5, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # (batch, H, W) or (batch,H,W,1)
    bce = tf.reduce_mean(bce, axis=list(range(1, len(bce.shape))))  # -> (batch,)

    dl = dice_loss(y_true, y_pred, smooth=smooth)  # -> (batch,)

    return bce_weight * bce + (1.0 - bce_weight) * dl

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

df = pd.read_csv("/home/mbouchou/airbus-ship-detection/masks_subset.csv")
img_ids = df["ImageId"].drop_duplicates().values
train_ids, val_ids = train_test_split(img_ids, test_size=0.2, random_state=42)

train_df = df[df["ImageId"].isin(train_ids)]
val_df   = df[df["ImageId"].isin(val_ids)]

train_gen = CachedNpyGenerator(train_df, CACHE_IMG_DIR, CACHE_MSK_DIR, batch_size=BATCH_SIZE, shuffle=True,  augment=True)
val_gen   = CachedNpyGenerator(val_df,   CACHE_IMG_DIR, CACHE_MSK_DIR, batch_size=BATCH_SIZE, shuffle=False, augment=False)

# Compile the model
seg_model.compile(
    optimizer=Adam(1e-4),
    loss=combo_loss,
    metrics=[dice_coef]
)


# Monitor validation dice coeff and save the best model weights
checkpoint = ModelCheckpoint(
    "seg_model_best.keras",
    monitor="val_dice_coef",
    mode="max",
    save_best_only=True,
    verbose=1
)

# Reduce Learning Rate on Plateau
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

# Stop training once there is no improvement seen in the model
early = EarlyStopping(monitor="val_dice_coef", 
                      mode="max", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited

# Callbacks ready
callbacks_list = [checkpoint, early, reduceLROnPlat]

steps_per_epoch = min(MAX_TRAIN_STEPS, len(train_gen))
validation_steps = len(val_gen)

history = seg_model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=NB_EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks_list
)

seg_model.save('seg_model.keras')