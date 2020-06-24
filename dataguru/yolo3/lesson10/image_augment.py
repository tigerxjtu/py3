import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import imageio
import numpy as np
import os
import sys
import cv2
import time


#图像增强之雪景变换
def imgaug_snowylandscape(images,val_thresh,mulfactor,base_save_path):
    fastsnow = iaa.FastSnowyLandscape(val_thresh,mulfactor)
    fastsnow_imgs = fastsnow.augment_images(images)

    fastsnow_path = '\\snowylandscape\\'
    if not os.path.exists(base_save_path+fastsnow_path):
        os.mkdir(base_save_path+fastsnow_path)

    name_index = 0
    for img in fastsnow_imgs:
        name_index += 1
        imageio.imwrite(base_save_path+fastsnow_path+'img_aug_snowlandscape_'+ time.strftime('%Y%m%d_%H',time.localtime()) \
                        + '_' +str(name_index)+'.jpg',img)

#图像增强之右转加雪
def imgaug_rightrotate_plus_snowflake(images,rotate_degree,base_save_path):
    # if  rotate_degree >= 0:
    #     rotate_degree = - rotate_degree
    right_rotate = iaa.Affine(rotate=rotate_degree)
    snowflake = iaa.Snowflakes((0.03, 0.06), (0.1, 0.5), (0.3, 0.6), (0.4, 0.8), (-30, 30), (0.007, 0.03))
    seq = iaa.Sequential([right_rotate,snowflake])
    dst_imgs = seq.augment_images(images)

    right_rotate_path = '\\r_rotate_plus_snow\\'
    if not os.path.exists(base_save_path + right_rotate_path):
        os.mkdir(base_save_path + right_rotate_path)

    name_index = 0
    for img in dst_imgs:
        name_index += 1
        imageio.imwrite(base_save_path + right_rotate_path + 'img_aug_r_rotate_snow_' + time.strftime('%Y%m%d_%H',time.localtime()) \
                        + '_' + str(name_index) + '.jpg', img)

#图像增强之左转加雪
def imgaug_leftrotate_plus_snowflake(images,rotate_degree,base_save_path):
    left_rotate = iaa.Affine(rotate=rotate_degree)
    snowflake = iaa.Snowflakes((0.01,0.04),(0.1, 0.5), (0.3, 0.6), (0.4, 0.8), (-30, 30), (0.007, 0.03))
    seq = iaa.Sequential([left_rotate,snowflake])
    dst_imgs = seq.augment_images(images)

    left_rotate_path = '\\l_rotate_plus_snow\\'
    if not os.path.exists(base_save_path + left_rotate_path):
        os.mkdir(base_save_path + left_rotate_path)

    name_index = 0
    for img in dst_imgs:
        name_index += 1
        imageio.imwrite(base_save_path + left_rotate_path + 'img_aug_l_rotate_snow_' + time.strftime('%Y%m%d_%H',time.localtime()) \
                        + '_' + str(name_index) + '.jpg', img)

#图像增强之加雪花
def imgaug_snowflake(images,base_save_path):
    snowflake = iaa.Snowflakes((0.01, 0.04), (0.1, 0.5), (0.3, 0.6), (0.4, 0.8), (-30, 30), (0.007, 0.03))
    snowflake_imgs = snowflake.augment_images(images)

    snowflake_path = '\\snowflake\\'
    if not os.path.exists(base_save_path + snowflake_path):
        os.mkdir(base_save_path + snowflake_path)

    name_index = 0
    for img in snowflake_imgs:
        name_index += 1
        imageio.imwrite(base_save_path + snowflake_path + 'img_aug_snowflake_' + time.strftime('%Y%m%d_%H',time.localtime()) \
                        + '_' + str(name_index) + '.jpg', img)

#图像增强之调节亮暗
def imgaug_darken(images,base_save_path):
    darken = iaa.WithColorspace(to_colorspace="HSV", children=iaa.WithChannels(2, iaa.Add((-60, -80))))
    darken_imgs = darken.augment_images(images)

    darken_path = '\\darken\\'
    if not os.path.exists(base_save_path + darken_path):
        os.mkdir(base_save_path + darken_path)

    name_index = 0
    for img in darken_imgs:
        name_index += 1
        imageio.imwrite(base_save_path + darken_path + 'img_aug_darken_' + time.strftime('%Y%m%d_%H',time.localtime()) \
                        + '_' + str(name_index) + '.jpg', img)

#图像增强之平移
def imgaug_translate(images,x_direct,y_direct,base_save_path):
    translate = iaa.Affine(translate_px={"x":x_direct,"y":y_direct})
    #snowflake = iaa.Snowflakes((0.03, 0.06), (0.1, 0.5), (0.3, 0.6), (0.4, 0.8), (-30, 30), (0.007, 0.03))
    seq = iaa.Sequential([translate])
    dst_imgs = seq.augment_images(images)

    translate_path = '\\translate\\'
    if not os.path.exists(base_save_path + translate_path):
        os.mkdir(base_save_path + translate_path)

    name_index = 0
    for img in dst_imgs:
        name_index += 1
        imageio.imwrite(base_save_path + translate_path + 'img_aug_translate_9_' + time.strftime('%Y%m%d_%H',time.localtime()) \
                        + '_' + str(name_index) + '.jpg', img)

#用来改变图像的通道数，有些图像通道数不等于3，该函数将所有图片通道数改为3，并将新图片覆盖旧图片
def change_img_channel(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("cv2.imread failed")
    else:
        if img.shape[2] != 3:
            b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
            g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
            r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
            b[:, :] = img[:, :, 0]
            g[:, :] = img[:, :, 1]
            r[:, :] = img[:, :, 2]

            merged_img = cv2.merge([b, g, r])
            print("******************")
            print("shape of img: " + img.shape)
            print("shape of merged_img" + merged_img.shape)
            print("******************")
            cv2.imwrite(img_path, merged_img)


if __name__ == '__main__':
    images = []

    images_read_path = os.getcwd() + '\\JPEGImages'
    files = os.listdir(images_read_path)
    print('The number of image which needs augment is ' + str(len(files)))
    for file in files:
        if file[-3:] == 'jpg':
            change_img_channel(images_read_path + '\\' + file)
            try:
                img = imageio.imread(images_read_path + '\\' + file)
            except:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('!!!!!!Picture: ' + file +' is not the right form!!!!!!!')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                continue
            images.append(img)
            print('Picture: ' + file + '  Loaded!')
    print('*************************')
    print('Picture Loading Done!')
    print('*************************')

    if not images:
        print("reading image failed，the path may be wrong!")
    else:
        base_save_path = os.getcwd() + '\\image_augment'
        if not os.path.exists(base_save_path):
            os.mkdir(base_save_path)

        #各个增强方法的参数作用具体见官方doc
        imgaug_darken(images, base_save_path)
        imgaug_rightrotate_plus_snowflake(images, (-25, -35), base_save_path)
        imgaug_leftrotate_plus_snowflake(images, 30, base_save_path)
        imgaug_snowylandscape(images, 80, 1.8, base_save_path)
        imgaug_translate(images, (0, 0), (1000, 1100), base_save_path)