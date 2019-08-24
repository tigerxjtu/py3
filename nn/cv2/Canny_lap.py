import cv2
import numpy as np
from scipy.ndimage import filters

img = cv2.imread("outline.png",0)

img = cv2.GaussianBlur(img,(3,3),0)
canny = cv2.Canny(img, 50, 150)
print(canny.shape,canny)

# gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
# print(gray_lap)
# dst = cv2.convertScaleAbs(gray_lap)

imx = np.zeros(canny.shape)
filters.sobel(canny,1,imx)
imy = np.zeros(canny.shape)
filters.sobel(canny,0,imy)
magnitude = np.sqrt(imx**2+imy**2)

result=magnitude[canny>0]
print(len(result))
print(result)
result=[result<1]
print(len(result))



# cv2.imshow('Canny',canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()