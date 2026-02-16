import numpy as np

def calculate_mse(image1, image2):
    diff = image1.astype(np.float32).flatten() - image2.astype(np.float32).flatten()
    res = np.mean(diff ** 2)
    return res

