import numpy as np
import histogram
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('Fig0326(a)(embedded_square_noisy_512).tif', cv.IMREAD_GRAYSCALE)

plt.imshow(img, cmap='gray')
hist = histogram.image_hist(img)
plt.figure()
plt.bar(np.arange(hist.shape[0]), hist)
plt.show()

#全局直方图均衡
global_hist_img = histogram.equalize_hist(img, hist)
plt.imshow(global_hist_img, cmap='gray')
hist = histogram.image_hist(global_hist_img)
plt.figure()
plt.bar(np.arange(hist.shape[0]), hist)
plt.show()

#局部直方图均衡
local_hist_img = histogram.local_equalize_hist(img, size=(100, 100))
plt.imshow(local_hist_img, cmap='gray')
hist = histogram.image_hist(local_hist_img)
plt.figure()
plt.bar(np.arange(hist.shape[0]), hist)
plt.show()

#局部直方图增强
img = cv.imread('Fig0327(a)(tungsten_original).tif', cv.IMREAD_GRAYSCALE)
local_enhance_img = histogram.local_enhance(img, 3, 3, 0, 0.25, 0, 0.1)
plt.imshow(np.hstack([img, local_enhance_img]), cmap='gray')
plt.show()