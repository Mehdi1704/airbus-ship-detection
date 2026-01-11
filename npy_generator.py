from tensorflow.keras.utils import Sequence
import numpy as np
import os
import math

class CachedNpyGenerator(Sequence):
    def __init__(self, csv_file, cache_img_dir, cache_msk_dir, batch_size=8, shuffle=True, augment=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        # one row per image
        self.image_ids = csv_file["ImageId"].drop_duplicates().values
        self.cache_img_dir = cache_img_dir
        self.cache_msk_dir = cache_msk_dir

        self.indexes = np.arange(len(self.image_ids))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_ids) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = self.image_ids[batch_indexes]

        X = np.empty((len(batch_ids), 256, 256, 3), dtype=np.float32)
        y = np.empty((len(batch_ids), 256, 256, 1), dtype=np.float32)

        for i, image_id in enumerate(batch_ids):
            img_path = os.path.join(self.cache_img_dir, image_id + ".npy")
            msk_path = os.path.join(self.cache_msk_dir, image_id + ".npy")

            img = np.load(img_path)  # float16 (256,256,3)
            msk = np.load(msk_path)  # uint8  (256,256,1)

            # cast to float32 for TF
            img = img.astype(np.float32)
            msk = msk.astype(np.float32)

            if self.augment:
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    msk = np.fliplr(msk)
                if np.random.rand() > 0.5:
                    img = np.flipud(img)
                    msk = np.flipud(msk)

            X[i] = img
            y[i] = msk

        return X, y