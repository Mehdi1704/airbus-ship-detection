import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import math

# Cluster Fix: Set backend to 'agg' before importing pyplot to avoid "no display" errors
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# --- Configuration ---
BATCH_SIZE = 16
IMAGE_SIZE = (256, 256)
EPOCHS = 20  # Increased epochs since we have early stopping
Train_v2_path = '/home/mbouchou/images'
Masks_csv_path = '/home/mbouchou/airbus-ship-detection/masks_subset.csv'

# --- Helper Functions ---
def rle_decode(mask_rle, shape=(768, 768)):
    if not isinstance(mask_rle, str):
        return np.zeros(shape)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def split_data(data, empty_masks=2000, test_size=0.3, random_state=42):
    masks_df = data.copy()
    masks_df['ship'] = masks_df['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    masks_df['n_ships'] = masks_df.groupby('ImageId')['ship'].transform('sum')
    masks_df.drop_duplicates(subset='ImageId', keep='first', inplace=True)
    
    # Balanced sampling
    empty_masks_df = masks_df[masks_df.ship == 0]
    masks_df = masks_df[masks_df.ship == 1]
    masks_df = pd.concat([masks_df, empty_masks_df.sample(n=empty_masks, random_state=random_state)], axis=0)
    
    train_ids, test_ids = train_test_split(masks_df, test_size=test_size, stratify=masks_df['n_ships'].values, random_state=random_state)
    return data[data['ImageId'].isin(train_ids.ImageId)], data[data['ImageId'].isin(test_ids.ImageId)]

# --- Custom Data Generator with Augmentation ---
class CustomDataGenerator(Sequence):
    def __init__(self, image_folder, csv_file, batch_size=32, image_size=(256, 256), shuffle=True, augment=False):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment  # New flag for augmentation
        
        self.image_ids = csv_file["ImageId"].drop_duplicates().values
        self.masks_by_image = csv_file.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()
        self.indexes = np.arange(len(self.image_ids))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_ids) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_ids = self.image_ids[batch_indexes]

        X = np.empty((len(batch_image_ids), self.image_size[0], self.image_size[1], 3), dtype=np.float32)
        y = np.empty((len(batch_image_ids), self.image_size[0], self.image_size[1], 1), dtype=np.float32)

        for i, image_id in enumerate(batch_image_ids):
            image_path = os.path.join(self.image_folder, image_id)
            img = cv2.imread(image_path)
            
            if img is None:
                # Handle missing images gracefully by creating a black image
                img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            
            img = img.astype(np.float32) / 255.0

            # Decode mask
            mask = np.zeros((768, 768), dtype=np.uint8)
            for m in self.masks_by_image.get(image_id, []):
                if isinstance(m, str):
                    mask += rle_decode(m)
            
            mask = (mask > 0).astype(np.uint8) # Ensure binary
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)

            # Apply Augmentation
            if self.augment:
                if np.random.rand() > 0.5: # Horizontal Flip
                    img = np.fliplr(img)
                    mask = np.fliplr(mask)
                if np.random.rand() > 0.5: # Vertical Flip
                    img = np.flipud(img)
                    mask = np.flipud(mask)
                if np.random.rand() > 0.5: # Rotate 90
                    k = np.random.randint(1, 4)
                    img = np.rot90(img, k=k)
                    mask = np.rot90(mask, k=k)

            X[i] = img
            y[i] = mask

        return X, y

# --- Load Data ---
train_segmentations = pd.read_csv(Masks_csv_path)
train_data, val_data = split_data(train_segmentations, empty_masks=2000, test_size=0.2)

# Enable augmentation only for training
train_generator = CustomDataGenerator(Train_v2_path, train_data, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=True, augment=True)
val_generator   = CustomDataGenerator(Train_v2_path, val_data,   batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=False, augment=False)

# --- U-Net Architecture ---
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 32) # Reduced filters slightly for speed/memory
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)
    
    b1 = conv_block(p4, 512)
    
    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)
    
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    return Model(inputs, outputs, name="U-Net")

# --- Custom Metrics and Loss ---
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def mixed_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_coefficient(y_true, y_pred)
    return bce + (1.0 - dice)

# --- Compile Model ---
model = build_unet(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), loss=mixed_loss, metrics=[dice_coefficient])
model.summary()

# --- Callbacks ---
# 1. Save Best Model
checkpoint = ModelCheckpoint('best_unet_model2.keras', monitor='val_dice_coefficient', mode='max', save_best_only=True, verbose=1)
# 2. Reduce LR if stuck
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
# 3. Stop early if no improvement
early_stopping = EarlyStopping(monitor="val_loss", patience=8, verbose=1, restore_best_weights=True)
# 4. Log history to CSV (Crucial for cluster!)
csv_logger = CSVLogger('training_history.csv', append=True)

callbacks_list = [checkpoint, reduce_lr, early_stopping, csv_logger]

# --- TF Data Pipeline ---
output_signature = (
    tf.TensorSpec(shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 1), dtype=tf.float32),
)

def make_gen(seq):
    def gen():
        for i in range(len(seq)):
            yield seq[i]
    return gen

train_ds = tf.data.Dataset.from_generator(
    make_gen(train_generator),
    output_signature=output_signature
).repeat() .prefetch(tf.data.AUTOTUNE)  # <--- .repeat() is mandatory!

val_ds = tf.data.Dataset.from_generator(
    make_gen(val_generator),
    output_signature=output_signature
).repeat() .prefetch(tf.data.AUTOTUNE)

# --- Training ---
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks_list,
    # --- ADD THESE LINES ---

)

# --- Saving Results ---
print("Training finished. Saving plots...")

# Plot Loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.legend()

# Plot Dice
plt.subplot(1, 2, 2)
plt.plot(history.history['dice_coefficient'], label='Train Dice')
plt.plot(history.history['val_dice_coefficient'], label='Val Dice')
plt.title('Dice Coefficient')
plt.xlabel('Epochs')
plt.legend()

plt.savefig('training_results2.png')
print("Plots saved to training_results2.png")