import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

train_segmentations = pd.read_csv('/home/mbouchou/airbus-ship-detection/masks_subset.csv')
## sample_submission = pd.read_csv('/kaggle/input/airbus-ship-detection/sample_submission_v2.csv')
train_v2_path = '/home/mbouchou/images'

epochs = 5

def rle_encode(img):
    """
    Encode a binary mask represented as a 2D numpy array using Run-Length Encoding (RLE).

    Parameters:
    - img (numpy.ndarray): A 2D binary array representing the mask.

    Returns:
    - str: The RLE-encoded string representing the binary mask.
    """

    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    """
    Decode a Run-Length Encoded (RLE) binary mask into a 2D numpy array.

    Parameters:
    - mask_rle (str): The RLE-encoded string representing the binary mask.
    - shape (tuple, optional): The shape of the target 2D array. Default is (768, 768).

    Returns:
    - numpy.ndarray: A 2D binary array representing the decoded mask.
    """

    if type(mask_rle) != str:
        return np.zeros(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T # Needed to align to RLE direction

def split_data(data, empty_masks=2000, test_size=0.3, random_state=42):
    """
    Parameters:
    - data (DataFrame): The input DataFrame containing the dataset.
    - empty_masks (int, optional): The number of images with empty masks. Defaults to 2000.
    - test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.3.
    - random_state (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 42.

    Returns: The training and testing sets.
    """

    masks_df = data.copy()

    # Create binary labels for the presence of ships in each image. Count the number of ships in each image.
    masks_df['ship'] = masks_df['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    masks_df['n_ships'] = masks_df.groupby('ImageId')['ship'].transform('sum')
    masks_df.drop_duplicates(subset='ImageId', keep='first', inplace=True)

    # Keep only n empty masks
    empty_masks_df = masks_df[masks_df.ship == 0]
    masks_df = masks_df[masks_df.ship == 1]
    masks_df = pd.concat([masks_df, empty_masks_df.sample(n=empty_masks, random_state=random_state)], axis=0)

    # Stratified split based on the number of ships in each image
    train_ids, test_ids = train_test_split(masks_df, test_size=test_size, stratify=masks_df['n_ships'].values,
                                           random_state=random_state)

    train_data = data[data['ImageId'].isin(train_ids.ImageId)]
    test_data = data[data['ImageId'].isin(test_ids.ImageId)]

    return train_data, test_data


train_data, val_data = split_data(train_segmentations, empty_masks=2000, test_size=0.2)

print(f'Number of masks in train data - {train_data.shape[0]}')
print(f'Number of masks in test data - {val_data.shape[0]}')

class CustomDataGenerator(Sequence):
    def __init__(self, image_folder, csv_file, batch_size=32, image_size=(256, 256), shuffle=True):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.image_size = image_size  # (H, W)
        self.shuffle = shuffle

        # One row per image for iteration
        self.image_ids = csv_file["ImageId"].drop_duplicates().values

        # Pre-group masks once (BIG speedup)
        self.masks_by_image = (
            csv_file.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()
        )

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

            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))  # cv2 expects (W,H)
            img = img.astype(np.float32) / 255.0
            X[i] = img

            mask = np.zeros((768, 768), dtype=np.uint8)
            for m in self.masks_by_image.get(image_id, []):
                if isinstance(m, str):
                    mask |= rle_decode(m, shape=(768, 768)).astype(np.uint8)

            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            y[i, ..., 0] = mask.astype(np.float32)

        return X, y
    
batch_size = 16
image_size = (256, 256)

train_generator = CustomDataGenerator(train_v2_path, train_data, batch_size=16, image_size=(256,256), shuffle=True)
val_generator   = CustomDataGenerator(train_v2_path, val_data,   batch_size=16, image_size=(256,256), shuffle=False)
output_signature = (
    tf.TensorSpec(shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, image_size[0], image_size[1], 1), dtype=tf.float32),
)

train_ds = tf.data.Dataset.from_generator(lambda: train_generator, output_signature=output_signature)
val_ds   = tf.data.Dataset.from_generator(lambda: val_generator,   output_signature=output_signature)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

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

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


# Create the UNet model
model = build_unet()

# Display the model summary
model.summary()



reduceLROnPlate = ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=1, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-8)
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=10)

callbacks_list = [reduceLROnPlate, early_stopping]

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(tf.abs(y_true * y_pred))
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

optimizer=tf.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=[dice_coefficient])


steps_per_epoch = len(train_generator)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    steps_per_epoch=len(train_generator),
    callbacks=callbacks_list,
    verbose=1,
)

plt.figure(figsize=(16, 5))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(range(epochs), history.history['loss'], 'bo-', label='Training loss')
plt.plot(range(epochs), history.history['val_loss'], 'ro-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation dice coefficient
plt.subplot(1, 2, 2)
plt.plot(range(epochs), history.history['dice_coefficient'], 'bo-', label='Training Dice Coefficient')
plt.plot(range(epochs), history.history['val_dice_coefficient'], 'ro-', label='Validation Dice Coefficient')
plt.title('Training and Validation Dice Coefficient')
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.legend()