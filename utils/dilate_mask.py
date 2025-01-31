import cv2
import numpy as np


mask = cv2.imread('mask.png', 0) 

kernel_size = 7
kernel = np.ones((kernel_size, kernel_size), np.uint8)

iterations = 10
dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)

eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)

cv2.imwrite('dilated_mask.png', eroded_mask)