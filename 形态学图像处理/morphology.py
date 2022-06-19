import numpy as np
import cv2 as cv

def hole_fill(img:np.ndarray, kernel:np.ndarray, start_pixel):
    last_temp_img = np.zeros_like(img)
    last_temp_img[start_pixel] = 1
    current_temp_img = cv.dilate(last_temp_img, kernel)
    while (current_temp_img != last_temp_img).any():
        last_temp_img = current_temp_img
        current_temp_img = cv.dilate(current_temp_img, kernel)
        current_temp_img = np.logical_and(current_temp_img == 1, current_temp_img == (1 - img)).astype(float)
    return current_temp_img