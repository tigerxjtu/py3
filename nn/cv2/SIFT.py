import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('outline.png')
SIFT = cv2.xfeatures2d.SIFT_create()
SURF = cv2.xfeatures2d.SURF_create()
ORB = cv2.ORB_create()

kps_SIFT,descfs_SIFT = SIFT.detectAndCompute(im, None)
kps_SURF,descfs_SURF = SURF.detectAndCompute(im, None)
kps_ORB, descfs_ORB = ORB.detectAndCompute(im, None)

imgSIFT = cv2.drawKeypoints(im, kps_SIFT, None, color=(255,0,0))
imgSURF = cv2.drawKeypoints(imgSIFT, kps_SURF, None, color=(0,255,0))
imgORB = cv2.drawKeypoints(imgSURF, kps_ORB, None, color=(0,0,255))

# Now we drawn the gray image and overlay the Key Points (kp)
img = cv2.drawKeypoints(im, kps_SIFT, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Plot it to the screen, looks a little small
plt.imshow(imgSIFT)
plt.show()