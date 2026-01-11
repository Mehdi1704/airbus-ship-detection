import numpy as np
# Load one of your training images manually
sample_img = np.load("/home/mbouchou/airbus-ship-detection-cache/images_256/0a0ada7e3.jpg.npy")
print(f"Max value in training data: {sample_img.min()}")