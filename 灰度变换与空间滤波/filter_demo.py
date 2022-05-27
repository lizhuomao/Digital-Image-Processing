import spatial_filter
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('./image/Fig0333(a)(test_pattern_blurring_orig).tif', cv.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')


#低通高斯滤波器
plt.figure()
img_gaussian_filtered = spatial_filter.Gaussian_filter(img, size=(21, 21))
plt.imshow(img_gaussian_filtered, cmap='gray')

plt.show()

#锐化空间滤波器 拉普拉斯核
img = cv.imread('./image/Fig0338(a)(blurry_moon).tif', cv.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
img_lapras_filtered = spatial_filter.Lapras_filter(img)

plt.figure()
plt.imshow(img_lapras_filtered, cmap='gray')
plt.figure()
plt.imshow(img - img_lapras_filtered, cmap='gray')
plt.show()