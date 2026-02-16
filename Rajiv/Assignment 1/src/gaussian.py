import numpy as np

def createGaussianKernel(P = 5, sigma = 3, normalized = True):
    win_size = (2*P) + 1
    kernel = np.zeros((win_size, win_size)).astype(np.float64)

    for i in range(win_size):
        for j in range(win_size):
            # finding the centered cords for gaussian calculation
            x = i - ((win_size - 1) // 2)
            y = j - ((win_size - 1) // 2)

            kernel[i,j] = np.exp(-((x**2) + (y**2)) / (2 * (sigma ** 2)))
    
    if normalized:
        kernel /= np.sum(kernel)
    return kernel