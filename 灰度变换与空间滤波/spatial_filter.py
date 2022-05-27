import numpy as np

def location_transform(c_position, total):
    if c_position < 0:
        return 0
    elif c_position > total:
        return total
    else:
        return c_position

def matrix_extension(matrix:np.ndarray, raw:int, col:int):
    matrix_t = np.zeros((matrix.shape[0] + raw * 2, matrix.shape[0] + col * 2))
    matrix_t[raw:raw + matrix.shape[0], col:col + matrix.shape[1]] = matrix.copy()
    return matrix_t


def scan_image(img:np.ndarray, operator:np.ndarray):
    h, w = img.shape[0], img.shape[1]
    mid_v = int(operator.shape[0] / 2)
    mid_h = int(operator.shape[1] / 2)
    img_t = matrix_extension(img, mid_v, mid_h)
    img_r = np.zeros(img.shape)
    for i in range(h):
        for j in range(w):
            img_r[i, j] = np.sum(np.multiply(img_t[i:i+operator.shape[0], j:j+operator.shape[1]], operator))
            # for m in range(operator.shape[0]):
            #     for n in range(operator.shape[1]):
            #         img_r[i, j] += img_t[i + m, j + n] * operator[m, n]
    return img_r

def Gaussian_filter(img:np.ndarray, size:(int, int), K=1):
    sigma = size[0] / 6
    operator = np.zeros(size)
    core = np.floor(size[0] / 2)
    _ = np.array([[0, 0, 0],[0, 1.0, 0],[0, 0, 0]])
    print(sigma, core)
    for i in range(size[0]):
        for j in range(size[1]):
            r2 = np.power(i - core, 2) + np.power(j - core, 2)
            operator[i, j] = K * np.exp(-r2 / (2 * np.power(sigma, 2)))
    return scan_image(img, operator)

def Lapras_filter(img:np.ndarray):
    operator = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return scan_image(img, operator)
