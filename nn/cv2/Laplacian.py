import cv2
import numpy as np
import matplotlib.pyplot as plt
# img = cv2.imread("outline.png", 0)
img = cv2.imread("201005100004_Front.jpg", 0)

gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
print(gray_lap)
dst = cv2.convertScaleAbs(gray_lap)
print(dst)
plt.imshow(dst,cmap='gray')
plt.show()
