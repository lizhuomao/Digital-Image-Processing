import numpy as np

def findZCP(img:np.ndarray, t, value):
    mid = (1, 1)
    img_t = np.zeros(((img.shape[0] + (mid[0] * 2)), (img.shape[1] + (mid[1] * 2))))
    img_t[mid[0]:img.shape[0] + mid[0], mid[1]:img.shape[1] + mid[1]] = img.copy()

    img_r = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cnt = 0
            if (img_t[i, j] * img_t[i + 2, j + 2]) < 0 and np.abs(img_t[i, j] - img_t[i + 2, j + 2]) > t:
                cnt = cnt + 1
            if img_t[i, j + 1] * img_t[i + 2, j + 1] < 0 and np.abs(img_t[i, j + 1] - img_t[i + 2, j + 1]) > t:
                cnt = cnt + 1
            if img_t[i, j + 2] * img_t[i + 2, j] < 0 and np.abs(img_t[i, j + 2] - img_t[i + 2, j]) > t:
                cnt = cnt + 1
            if img_t[i + 1, j] * img_t[i + 1, j + 2] < 0 and np.abs(img_t[i + 1, j] - img_t[i + 1, j + 2]) > t:
                cnt = cnt + 1
            if cnt >= 2:
                img_r[i, j] = value
            else:
                img_r[i, j] = 0
    return img_r