import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

img = cv2.imread('body.png')
print(img.shape)
(height,width,deep) = img.shape

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dest_img = np.zeros((height,width),np.uint8)

Sobelx = [[1,2,1],[0,0,0],[-1,-2,-1]]
Sobely = [[1,0,-1],[2,0,-2],[1,0,-1]]
Prewittx = [[-1,0,1],[-1,0,1],[-1,0,1]]
Prewitty = [[-1,-1,-1],[0,0,0],[1,1,1]]
Sobelxim = signal.convolve2d(gray,Sobelx,boundary='symm', mode='same')
Sobelyim = signal.convolve2d(gray,Sobely,boundary='symm', mode='same')
Prewittxim = signal.convolve2d(gray,Prewittx,boundary='symm', mode='same')
Prewittyim = signal.convolve2d(gray,Prewitty,boundary='symm', mode='same')
Sobelximabs = np.abs(Sobelyim)
edge_cut = np.floor((Sobelximabs - np.min(Sobelximabs)) * 255 /((np.max(Sobelximabs) - np.min(Sobelximabs))))

plt.imshow(edge_cut,cmap='gray')
plt.colorbar()
plt.show()