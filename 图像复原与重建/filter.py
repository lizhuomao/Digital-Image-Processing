import numpy as np

def arithmetic_average_filter(img:np.ndarray, m:int, n:int):
    mid = (m // 2, n // 2)
    img_t = np.zeros(((img.shape[0] + (mid[0] * 2)), (img.shape[1] + (mid[1] * 2))))
    img_t[mid[0]:img.shape[0] + mid[0], mid[1]:img.shape[1] + mid[1]] = img

    img_r = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            g = 0
            for l in np.arange(m):
                for k in np.arange(n):
                    g += img_t[i + l, j + k]
            img_r[i, j] = g / (m * n)
    return img_r

def Geometric_mean_filter(img:np.ndarray, m:int, n:int):
    mid = (m // 2, n // 2)
    img_t = np.ones(((img.shape[0] + (mid[0] * 2)), (img.shape[1] + (mid[1] * 2))))
    img_t[mid[0]:img.shape[0] + mid[0], mid[1]:img.shape[1] + mid[1]] = img

    img_r = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            g = 1
            for l in np.arange(m):
                for k in np.arange(n):
                    g *= img_t[i + l, j + k]
            img_r[i, j] = np.power(g, 1 / (m * n))
    return img_r

def maximum_filter(img:np.ndarray, m:int, n:int):
    mid = (m // 2, n // 2)
    img_t = np.zeros(((img.shape[0] + (mid[0] * 2)), (img.shape[1] + (mid[1] * 2))))
    img_t[mid[0]:img.shape[0] + mid[0], mid[1]:img.shape[1] + mid[1]] = img

    img_r = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            g = 0
            for l in np.arange(m):
                for k in np.arange(n):
                    if g < img_t[i + l, j + k]:
                        g = img_t[i + l, j + k]
            img_r[i, j] = g
    return img_r

def butterworth_filter(img:np.ndarray, D0:int, uvk:np.ndarray, n:int):
    H = np.empty_like(img, float)
    m, n = img.shape[0] // 2, img.shape[1] // 2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            H[y, x] = 1
            for uk, vk in uvk:
                D = np.sqrt(np.power(x - n - uk, 2) + np.power(y - m - vk))
                Di = np.sqrt(np.power(x - n + uk, 2) + np.power(y - m + vk))
                H[y, x] *= (1 / (1 + np.power(D0 / (D + 1e-6)), n)) * (1 / (1 + np.power(D0 / (Di + 1e-6), n)))
    return H
