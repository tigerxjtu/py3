import cv2
import numpy as np

target_cols,target_rows = 608,608

img = np.ones((target_rows,target_cols,3),dtype=np.uint8)
img[:,:,:] = 127

src = cv2.imread('dog_bike_car.jpg')
print(src.shape,img.shape)

def get_new_size(src_cols,src_rows, target_cols,target_rows):
    if float(target_cols)/src_cols < float(target_rows)/src_rows:
        new_cols = target_cols
        new_rows = (float(target_cols)/src_cols) * src_rows
    else:
        new_rows = target_rows
        new_cols = (float(target_rows)/src_rows) * src_cols
    return int(new_cols),int(new_rows)

new_cols,new_rows = get_new_size(src.shape[1],src.shape[0],target_cols,target_rows)
src_img = cv2.resize(src,(new_cols,new_rows))

x=int((target_rows-new_rows)/2)
y=int((target_cols-new_cols)/2)

print(x,y,new_cols,new_rows)

img[x:x+new_rows,y:y+new_cols,:]=src_img

cv2.imshow('img',img)
cv2.waitKey(0)
