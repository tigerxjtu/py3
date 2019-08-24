import cv2
import numpy as np
import matplotlib.pyplot as plt

im1 = cv2.imread('White_House_Images/whitehouse1.jpg') #read the picture
im2 = cv2.imread('White_House_Images/whitehouse2.jpg')

SIFT_1 = cv2.xfeatures2d.SIFT_create() #sift create
SIFT_2 = cv2.xfeatures2d.SIFT_create() #sift create

kps_SIFT_1, desc_SIFT_1 = SIFT_1.detectAndCompute(im1, None) #detect and compute the key points and descriptors
kps_SIFT_2, desc_SIFT_2 = SIFT_2.detectAndCompute(im2, None) #detect and compute the key points and descriptors

im_dest_1  = cv2.drawKeypoints(im1, kps_SIFT_1, (0,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_dest_2  = cv2.drawKeypoints(im2, kps_SIFT_2, (0,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# create a BFmacher object

bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck = True)

matches = bf.match(desc_SIFT_1,desc_SIFT_2)

matches = sorted(matches, key = lambda x:x.distance)

# first 100 matches


Num_Matches = 1000

match_img = cv2.drawMatches(
    im1,kps_SIFT_1,
    im2,kps_SIFT_2,
    matches[: Num_Matches],None,flags = 0

)


plt.imshow(match_img)
plt.show()
