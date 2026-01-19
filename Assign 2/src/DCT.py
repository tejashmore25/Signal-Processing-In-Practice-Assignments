import numpy as np

def generateDCTbasis(k, n, N):
    if k == 0:
        const = np.sqrt(1 / N)
    else:
        const = np.sqrt(2 / N)
    value = const * np.cos((np.pi * (n + 0.5) * k) / N)
    return value

def generateDCTBasis2D(u, v, M, N):
    DCT = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            phi_1 = generateDCTbasis(u, m, M)
            phi_2 = generateDCTbasis(v, n, N)
            DCT[m, n] = phi_1 * phi_2
    return DCT

def computeDCTcoeff(image, DCT_basis_2D):
    M, N = image.shape
    img_DCT = np.zeros((M, N))

    for u in range(M):
        for v in range(N):
            DCT_basis = DCT_basis_2D[u, v]
            x_hat = np.sum(image * DCT_basis)
            img_DCT[u,v] = x_hat
    return img_DCT

def generateQ(s, M, N):
    Q = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            Q[u, v] = 1 + s * (u + v)
    return Q

def reconstructImage(img_DCT, DCT_basis_2D, QuantizeStrenth = 0):  
    M, N = img_DCT.shape
    recon_img = np.zeros((M, N))
    if QuantizeStrenth:
        Q = generateQ(QuantizeStrenth, M, N)
        Q_block = np.round(img_DCT / Q)
        recon_coeff = Q_block * Q
    else:
        recon_coeff = img_DCT

    for m in range(M):
        for n in range(N):               
            x = recon_coeff[m, n] * DCT_basis_2D[m, n]
            recon_img += x

    return recon_img, recon_coeff

def quantizeImage(img, DCT_basis_2D, compressionStrength = 0):
    if img.shape[0] % 8 != 0 or img.shape[1] % 8 != 0:
        print("Image dimension must be divisible by 8")
        return
    
    M, N = img.shape
    img_DCT = np.zeros((M, N))
    recon_img = np.zeros((M, N))
    recon_coef = np.zeros((M, N))

    for m in range(0, M, 8):
        for n in range(0, N, 8):
            img_DCT[m:m+8, n:n+8] = computeDCTcoeff(img[m:m+8, n:n+8], DCT_basis_2D)
            recon_img[m:m+8, n:n+8], recon_coef[m:m+8, n:n+8] = reconstructImage(img_DCT[m:m+8, n:n+8], DCT_basis_2D, QuantizeStrenth = compressionStrength)
    return recon_img, img_DCT, recon_coef

def computeMAE(img, recon_img):
    error = np.max(np.abs(img - recon_img))
    return error

def computeMSE(img, recon_img):
    return np.sum((img - recon_img) ** 2) / img.size

def computePSNR(img, recon_img):
    mse = computeMSE(img, recon_img)
    peak = 255
    psnr = 10 * np.log10(((peak) ** 2) / mse)
    return psnr

def computeSparsity(img):
    return np.sum(img == 0) / img.size
