import numpy as np

def smoothing_RGB(img:np.ndarray, m, n):
    mid = (m // 2, n // 2)
    img_t = np.zeros(((img.shape[0] + (mid[0] * 2)), (img.shape[1] + (mid[1] * 2)), img.shape[2]))
    img_t[mid[0]:img.shape[0] + mid[0], mid[1]:img.shape[1] + mid[1]] = img.copy()

    img_r = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for p in range(img.shape[2]):
                g = 0
                for l in np.arange(m):
                    for k in np.arange(n):
                        g += img_t[i + l, j + k, p]
                img_r[i, j, p] = g / (m * n)
    return img_r

def smoothing_HSI(img:np.ndarray, m, n):
    mid = (m // 2, n // 2)
    img_t = np.zeros(((img.shape[0] + (mid[0] * 2)), (img.shape[1] + (mid[1] * 2)), img.shape[2]))
    img_t[mid[0]:img.shape[0] + mid[0], mid[1]:img.shape[1] + mid[1]] = img.copy()

    img_r = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            g = 0
            for l in np.arange(m):
                for k in np.arange(n):
                    g += img_t[i + l, j + k, 2]
            img_r[i, j, 2] = g / (m * n)
    return img_r

def sharpening_RGB(img:np.ndarray):
    lapras = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    mid = (1, 1)
    img_t = np.zeros(((img.shape[0] + (mid[0] * 2)), (img.shape[1] + (mid[1] * 2)), img.shape[2]))
    img_t[mid[0]:img.shape[0] + mid[0], mid[1]:img.shape[1] + mid[1]] = img.copy()

    img_r = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for p in range(img.shape[2]):
                g = 0
                for l in range(lapras.shape[0]):
                    for k in range(lapras.shape[1]):
                      g += (img_t[i + l, j + k, p] * lapras[l, k])
                img_r[i, j, p] = g
    return img_r

def sharpening_HSI(img:np.ndarray):
    lapras = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    mid = (1, 1)
    img_t = np.zeros(((img.shape[0] + (mid[0] * 2)), (img.shape[1] + (mid[1] * 2)), img.shape[2]))
    img_t[mid[0]:img.shape[0] + mid[0], mid[1]:img.shape[1] + mid[1]] = img.copy()

    img_r = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            g = 0
            for l in range(lapras.shape[0]):
                for k in range(lapras.shape[1]):
                    g += (img_t[i + l, j + k, 2] * lapras[l, k])
            img_r[i, j, 2] = g
    return img_r