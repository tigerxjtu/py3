import numpy as np
import cv2

def load_img_gray(img_file, size=(8,8)):
    img = cv2.imread(img_file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img_gray,size)

def _avg_hash(img):
    array = img.flatten()
    mean_value = np.average(array)
    np_func = np.frompyfunc(lambda x: 1 if x>mean_value else 0,1,1)
    return np_func(array)

def avg_hash(img_file):
    img = load_img_gray(img_file)
    return _avg_hash(img)

def dct_hash(img_file):
    img = load_img_gray(img_file, size=(32, 32))
    img = np.float32(img)
    img_dct = cv2.dct(img)
    return _avg_hash(img_dct[:8,:8])

def d_hash(img_file):
    img = load_img_gray(img_file, size=(9, 8))
    img = img.astype(np.int32)
    img1 = img[:, :8]
    img2 = img[:, 1:]
    diff = img1 - img2
    np_func = np.frompyfunc(lambda x: 1 if x > 0 else 0, 1, 1)
    return np_func(diff).flatten()

def _hemin_distance(hash1, hash2):
    diff = np.abs(hash1-hash2)
    return np.sum(diff)

def hemin_distance(img_file1,img_file2, hash_func=avg_hash):
    hash1=hash_func(img_file1)
    hash2=hash_func(img_file2)
    return _hemin_distance(hash1,hash2)

if __name__ == '__main__':
    file = 'trunk1.jpg'
    # img = load_img_gray(file)
    # print(avg_hash(img))

    # img = load_img_gray(file,size=(32,32))
    # img = np.float32(img)
    # img_dct = cv2.dct(img)
    # print(img_dct[:8,:8])

    # img = load_img_gray(file, size=(9, 8))
    # print(img.shape)
    # img = img.astype(np.int32)
    # img1 = img[:,:8]
    # img2 = img[:,1:]
    # diff = img1-img2
    # print(img1,img2)
    # print(diff.shape,diff)

    file1= 'trunk1.jpg'
    file2='trunk2.jpg'

    print(hemin_distance(file1,file2))
    print(hemin_distance(file1, file2, dct_hash))
    print(hemin_distance(file1, file2, d_hash))

    print(hemin_distance(file1,file1))
    print(hemin_distance(file2, file2, dct_hash))
    print(hemin_distance(file1, file1, d_hash))

