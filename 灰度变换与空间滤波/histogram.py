import numpy as np

def image_hist(img: np.ndarray):
    hist = np.zeros(img.max() + 1)
    img_f = img.flatten()
    for i in range(img_f.shape[0]):
        hist[img_f[i]] += 1
    return hist

def equalize_hist(img:np.ndarray, hist:np.ndarray):
    pr = hist / hist.sum()
    s = np.zeros(pr.shape[0])
    for i in range(s.shape[0]):
        s[i] = img.max() * pr[:i + 1].sum()
    global_hist_img = s[img]
    return global_hist_img.astype(int)

def location_transform(c_position, total):
    if c_position < 0:
        return 0
    elif c_position > total:
        return total - 1
    else:
        return c_position

def is_shadow(local_mean, local_std, global_mean, global_std, k0, k1, k2, k3):
    temp1 = k0 * global_mean <= local_mean <= k1 * global_mean
    temp2 = k2 * global_std <= local_std <= k3 * global_std
    return temp1 and temp2

def get_dark_area(img: np.ndarray, C, local_size, k0, k1, k2, k3):
    global_mean = np.mean(img.flatten())
    global_std = np.std(img.flatten())

    shadow_matrix = np.zero((img.shape[0], img.shape[1]))
    h, w = img.shape
    local_half = (local_size / 2.0).astype(int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            start_i = location_transform(i - local_half, h)
            end_i = location_transform(i + local_half, h)
            start_j = location_transform(j - local_half, w)
            end_j = location_transform(j + local_half, w)


            local_mean = np.mean(img[start_i:end_i, start_j:end_j].flatten())
            local_std = np.std(img[start_i:end_i, start_j:end_j].flatten())
            shadow_matrix[i][j] = is_shadow(local_mean, local_std ,global_mean, global_std, k0, k1, k2, k3)
    return shadow_matrix


def local_enhance(img: np.ndarray, C, local_size=3, k0=0, k1=1, k2=0, k3=1):
    shadow_matrix = get_dark_area(img, local_size, k0, k1, k2, k3)

    temp1 = img * shadow_matrix * C
    temp2 = np.logical_not(shadow_matrix)
    temp2 = temp2.astype(int)
    temp2 = img * temp2

    return temp1 + temp2
