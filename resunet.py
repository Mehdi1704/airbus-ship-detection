import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import math

# Force non-interactive backend for cluster environment
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

# Install: pip install segmentation-models
import segmentation_models as sm
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Set framework to Keras/TF
sm.set_framework('tf.keras')

# --- Configuration ---
BATCH_SIZE = 8  # Reduced for ResNet152/101 memory requirements
IMAGE_SIZE = (256, 256)
EPOCHS = 50
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

def split_data(data, empty_masks=2000, test_size=0.2, random_state=42):
    masks_df = data.copy()
    masks_df['ship'] = masks_df['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    masks_df['n_ships'] = masks_df.groupby('ImageId')['ship'].transform('sum')
    masks_df.drop_duplicates(subset='ImageId', keep='first', inplace=True)
    
    empty_masks_df = masks_df[masks_df.ship == 0]
    masks_df = masks_df[masks_df.ship == 1]
    masks_df = pd.concat([masks_df, empty_masks_df.sample(n=empty_masks, random_state=random_state)], axis=0)
    
    train_ids, test_ids = train_test_split(masks_df, test_size=test_size, stratify=masks_df['n_ships'].values, random_state=random_state)
    return data[data['ImageId'].isin(train_ids.ImageId)], data[data['ImageId'].isin(test_ids.ImageId)]

# --- Data Generator ---
class CustomDataGenerator(Sequence):
    def __init__(self, image_folder, csv_file, batch_size=32, image_size=(256, 256), shuffle=True, augment=False):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
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

        X = np.empty((len(batch_image_ids), *self.image_size, 3), dtype=np.float32)
        y = np.empty((len(batch_image_ids), *self.image_size, 1), dtype=np.float32)

        for i, image_id in enumerate(batch_image_ids):
            image_path = os.path.join(self.image_folder, image_id)
            img = cv2.imread(image_path)
            if img is None:
                img = np.zeros((*self.image_size, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            
            # ResNet specific preprocessing
            img = sm.get_preprocessing('resnet101')(img)

            mask = np.zeros((768, 768), dtype=np.uint8)
            for m in self.masks_by_image.get(image_id, []):
                if isinstance(m, str):
                    mask += rle_decode(m)
            
            mask = (mask > 0).astype(np.uint8)
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)

            if self.augment:
                if np.random.rand() > 0.5:
                    img, mask = np.fliplr(img), np.fliplr(mask)
                if np.random.rand() > 0.5:
                    img, mask = np.flipud(img), np.flipud(mask)

            X[i], y[i] = img, mask
        return X, y

# --- Model Creation ---
# Backbone choice: resnet101 is very deep and powerful
BACKBONE = 'resnet101'
model = sm.UnetPlusPlus(
    BACKBONE, 
    encoder_weights='imagenet', 
    classes=1, 
    activation='sigmoid', 
    input_shape=(256, 256, 3)
)

# Focal + Dice Loss: The breakthrough for 0.55 score
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1.0 * focal_loss)

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=1e-4),
    loss=total_loss,
    metrics=[sm.metrics.iou_score]
)

# --- Pipeline ---
train_df, val_df = split_data(pd.read_csv(Masks_csv_path))
train_gen = CustomDataGenerator(Train_v2_path, train_df, batch_size=BATCH_SIZE, augment=True)
val_gen = CustomDataGenerator(Train_v2_path, val_df, batch_size=BATCH_SIZE, augment=False, shuffle=False)

def tf_dataset(gen):
    dataset = tf.data.Dataset.from_generator(
        lambda: (gen[i] for i in range(len(gen))),
        output_signature=(
            tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32)
        )
    )
    return dataset.repeat().prefetch(tf.data.AUTOTUNE)

# --- Callbacks ---
callbacks = [
    ModelCheckpoint('best_resnet_unetpp.keras', monitor='val_iou_score', mode='max', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    CSVLogger('training_log_v3.csv')
]

# --- Train ---
history = model.fit(
    tf_dataset(train_gen),
    validation_data=tf_dataset(val_gen),
    epochs=EPOCHS,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=callbacks
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

plt.savefig('training_results4.png')
print("Plots saved to training_results4.png")