import numpy as np

def getabite(data:int, a:int):
    bite = data & (np.power(2, a))
    return bite

def getbiteplane(img:np.ndarray, a:int):
    bite = img & (np.power(2, a))
    return bite

def graycode(img:np.ndarray, m:int):
    img_g = np.zeros_like(img)
    for a in range(m):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                g = 0
                if a != m - 1:
                    g = getabite(img[i, j], a) ^ getabite(img[i, j], a + 1)
                else:
                    g = getabite(img[i, j], a)
                img_g[i, j] += g
    return img_g