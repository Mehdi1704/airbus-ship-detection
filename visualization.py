import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import tensorflow as tf
from keras.models import load_model
from skimage.measure import label, regionprops

folder_path='/Users/mbouchou/Downloads/airbus-ship-detection/test_v2/'
img_name = '00a3ab3cc.jpg'


def preprocess_input(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        return None  # signal "skip this file"

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

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

def rle_encode(mask: np.ndarray) -> str:
    """
    Encode a binary mask into RLE, matching this decode:
        return img.reshape(shape).T

    mask: 2D array (H, W), values {0,1} or {False,True}
    returns: RLE string (start length start length ...)
    """
    if mask is None:
        return ""

    # Ensure 2D binary
    mask = (mask > 0).astype(np.uint8)

    # IMPORTANT: transpose to match your rle_decode (.T)
    pixels = mask.T.flatten()

    # Add sentinels at both ends
    pixels = np.concatenate([[0], pixels, [0]])

    # Find run starts/ends
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def mixed_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_coefficient(y_true, y_pred)
    return bce + (1.0 - dice)

model = load_model(
    "/Users/mbouchou/Desktop/airbus-ship-detection/best_unet_model.keras", compile=False)  # or .h5
    #custom_objects={"dice_coefficient": dice_coefficient}
#)
img_path = os.path.join(folder_path, img_name)

x = preprocess_input(img_path)
if x is None:
    raise FileNotFoundError(img_path)

t0 = time.time()
pred = model(x, training=False).numpy()
print(f"Predicted batch of {len(x)} in {time.time() - t0:.2f}s")
pred2d = np.squeeze(pred)                        # (256,256) or (256,256,1)
if pred2d.ndim == 3:
    pred2d = pred2d[..., 0]                      # -> (256,256)

mask2d = (pred2d > 0.3).astype(np.uint8)         # 2D binary
img = np.squeeze(x, axis=0)                      # (256,256,3)



def make_submission(folder_path, model, threshold=0.3, batch_size=1):
    list_of_images = sorted(os.listdir(folder_path))  # debug: first 50 only
    image_id = []
    encoded_pixels = []

    batch_names = []
    batch_imgs = []

    def flush_batch():
        nonlocal batch_names, batch_imgs, image_id, encoded_pixels
        if not batch_imgs:
            return

        X = np.concatenate(batch_imgs, axis=0)  # (B,256,256,3)
        preds = model(X, training=False).numpy()  # (B,256,256,1)

        for img_name, pred in zip(batch_names, preds):
            pred2d = np.squeeze(pred)
            if pred2d.ndim == 3:
                pred2d = pred2d[..., 0]

            # resize probabilities explicitly
            pred2d = cv2.resize(pred2d.astype(np.float32), (768, 768), interpolation=cv2.INTER_LINEAR)
            mask = (pred2d > threshold).astype(np.uint8)

            if mask.sum() == 0:
                image_id.append(img_name)
                encoded_pixels.append("")
                continue

            labeled_mask = label(mask)
            regions = list(regionprops(labeled_mask))

            # guarantee at least one row per image
            if len(regions) == 0:
                image_id.append(img_name)
                encoded_pixels.append("")
                continue

            for region in regions:
                single_ship_mask = (labeled_mask == region.label).astype(np.uint8)
                image_id.append(img_name)
                encoded_pixels.append(rle_encode(single_ship_mask))

        batch_names, batch_imgs = [], []

    for img_name in list_of_images:
        x = preprocess_input(os.path.join(folder_path, img_name))
        if x is None:
            continue  # skip non-images / unreadables

        batch_names.append(img_name)
        batch_imgs.append(x)

        if len(batch_imgs) >= batch_size:
            flush_batch()

    flush_batch()

    return pd.DataFrame({"ImageId": image_id, "EncodedPixels": encoded_pixels})


submission = make_submission(folder_path, model)    
submission.to_csv("submission2.csv", index=False)
print("Submission file created: submission.csv")
print(submission.head())

one_sample_masks = submission[submission.ImageId == img_name].reset_index(drop=True)
''' Visualize results
for k in range(3):
    plt.subplot(2, 3, 4 + k)
    if k < len(one_sample_masks) and isinstance(one_sample_masks.EncodedPixels.iloc[k], str) and one_sample_masks.EncodedPixels.iloc[k] != "":
        plt.imshow(rle_decode(one_sample_masks.EncodedPixels.iloc[k]))
        plt.title(f"Decoded Mask {k+1}")
    else:
        plt.imshow(np.zeros((768,768), dtype=np.uint8))
        plt.title(f"Decoded Mask {k+1} (none)")
    plt.axis("off")
     '''