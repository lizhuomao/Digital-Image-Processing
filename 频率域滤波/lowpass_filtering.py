import numpy as np

#高斯低通滤波器
def GLPF(img:np.ndarray, D0:int):
    H = np.empty_like(img, float)
    m, n = img.shape[0] / 2, img.shape[1] / 2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            D = np.sqrt(np.power(x - n, 2) + np.power(y - m, 2))
            H[y, x] = np.exp(-np.power(D, 2) / (2 * np.power(D0, 2)))
    return H

def ILPF(img:np.ndarray, D0:int):
    H = np.zeros_like(img, float)
    m, n = img.shape[0] / 2, img.shape[1] / 2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            D = np.sqrt(np.power(x - n, 2) + np.power(y - m, 2))
            if D <= D0: H[y, x] = 1
            else: H[y, x] = 0
    return H

def BLPF(img:np.ndarray, D0:int, n:int):
    H = np.empty_like(img, float)
    m, n = img.shape[0] // 2, img.shape[1] // 2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            D = np.sqrt(np.power(x - n, 2) + np.power(y - m, 2))
            H[y, x] = 1 / (1 + np.power((D + 1e-6) / D0, 2 * n))
    return H
