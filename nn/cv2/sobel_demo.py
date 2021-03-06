from PIL import Image
import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import cv2
# im = np.array(Image.open('outline.png').convert('L'))
im=cv2.imread('outline.png',0)
#Sobel derivative filters
imx = np.zeros(im.shape)
filters.sobel(im,1,imx)
imy = np.zeros(im.shape)
filters.sobel(im,0,imy)
magnitude = np.sqrt(imx**2+imy**2)
magnitude_inverte = 255 - magnitude
plt.imshow(magnitude_inverte,cmap='gray')
plt.show()