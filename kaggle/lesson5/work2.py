import numpy as np
import cv2

def hist_inter(a, b, bins, thresh=0):
    # Uses Histogram intersection distance function to measure instability

    # Find range
    hist_max = max(max(a), max(b))
    hist_min = min(min(a), min(b))

    # np.histogram 'normed' is broken, normalisation must be done manually
    hist_a = np.histogram(a, bins=bins, range=(hist_min, hist_max), normed=False)[0].tolist()
    hist_b = np.histogram(b, bins=bins, range=(hist_min, hist_max), normed=False)[0].tolist()

    # Manual normalisation of histograms
    size_a = len(a)
    size_b = len(b)
    hist_a = [x / size_a for x in hist_a]
    hist_b = [x / size_b for x in hist_b]

    k = 0
    i = 0
    # Evaluate histogram intersection
    for d in zip(hist_a, hist_b):
        if sum(d) > thresh:
                k += min(d)
                i += 1

    return k
#
# def hist_inter2(a, b, bins, thresh=0):
#     hist_max = max(max(a), max(b))
#     hist_min = min(min(a), min(b))
#
#     # np.histogram 'normed' is broken, normalisation must be done manually
#     hist_a = np.histogram(a, bins=bins, range=(hist_min, hist_max), density=True)[0].tolist()
#     hist_b = np.histogram(b, bins=bins, range=(hist_min, hist_max), density=True)[0].tolist()
#
#     k = 0
#     i = 0
#     # Evaluate histogram intersection
#     for d in zip(hist_a, hist_b):
#         if sum(d) > thresh:
#             k += min(d)
#             i += 1
#
#     return k

def load_img_gray(img_file):
    img = cv2.imread(img_file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def img_hist_inter(img_file1,img_file2):
    img1= load_img_gray(img_file1)
    img2=load_img_gray(img_file2)
    return hist_inter(img1.flatten(),img2.flatten(),20)


if __name__ == '__main__':
    file1 = 'trunk1.jpg'
    file2 = 'trunk2.jpg'
    print(img_hist_inter(file1,file2))
