import numpy as np


#高斯低通滤波器
def GHPF(img:np.ndarray, D0:int):
    H = np.empty_like(img, float)
    m, n = img.shape[0] / 2, img.shape[1] / 2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            D = np.sqrt(np.power(x - n, 2) + np.power(y - m, 2))
            H[y, x] = np.exp(-np.power(D, 2) / (2 * np.power(D0, 2)))
    return 1 - H

def IHPF(img:np.ndarray, D0:int):
    H = np.zeros_like(img, float)
    m, n = img.shape[0] / 2, img.shape[1] / 2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            D = np.sqrt(np.power(x - n, 2) + np.power(y - m, 2))
            if D <= D0: H[y, x] = 1
            else: H[y, x] = 0
    return 1 - H

def BHPF(img:np.ndarray, D0:int, n:int):
    H = np.empty_like(img, float)
    m, n = img.shape[0] / 2, img.shape[1] / 2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            D = np.sqrt(np.power(x - n, 2) + np.power(y - m, 2))
            H[y, x] = 1 / (1 + np.power(D / D0, 2 * n))
    return 1 - H



def equalize_hist(img:np.ndarray, hist:np.ndarray):
    pr = hist / hist.sum()
    s = np.zeros(pr.shape[0])
    for i in range(s.shape[0]):
        s[i] = img.max() * pr[:i + 1].sum()
    global_hist_img = s[img]
    return global_hist_img.astype(int)

def image_hist(img: np.ndarray):
    hist = np.zeros(img.max() + 1)
    img_f = img.flatten()
    for i in range(img_f.shape[0]):
        hist[img_f[i]] += 1
    return hist