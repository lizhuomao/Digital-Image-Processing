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

global_hist_img = histogram.equalize_hist(img, hist)
plt.imshow(global_hist_img, cmap='gray')
hist = histogram.image_hist(global_hist_img)
plt.figure()
plt.bar(np.arange(hist.shape[0]), hist)
plt.show()