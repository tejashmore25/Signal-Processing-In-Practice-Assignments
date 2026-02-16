import numpy as np
from .gaussian import *

def bilateralFilter(image, P = 5, gaussian_sigma = 3, intensity_sigma = 3):
    gaussian_kernel = createGaussianKernel(P, gaussian_sigma, normalized=False)
    image = image.astype(np.float64)
    height, width = image.shape

    num = np.zeros_like(image)
    den = np.zeros_like(image)

    for k in range(-P, P + 1):
        for l in range(-P, P + 1):
            G = gaussian_kernel[k+P, l+P]

            row_start_neig, row_end_neigh = max(0, k), min(height, height + k)
            col_start_neig, col_end_neigh = max(0, l), min(width, width + l)
            row_start_center, row_end_center = max(0, -k), min(height, height - k)
            col_start_center, col_end_center = max(0, -l), min(width, width - l)

            neigh_pixels = image[row_start_neig:row_end_neigh, col_start_neig:col_end_neigh]
            center_pixels = image[row_start_center:row_end_center, col_start_center:col_end_center]

            diff = center_pixels - neigh_pixels
            H = np.exp(-(diff ** 2) / (2 * (intensity_sigma ** 2)))
            weight = G * H

            num[row_start_center:row_end_center, col_start_center: col_end_center] += (weight * neigh_pixels)
            den[row_start_center:row_end_center, col_start_center: col_end_center] += weight
        
    return num / den


