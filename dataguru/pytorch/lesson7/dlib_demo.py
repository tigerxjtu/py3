import dlib
from skimage import io
import cv2

# 使用特征提取器frontal_face_detector
detector = dlib.get_frontal_face_detector()

# path是图片所在路径
path = "c:\\tmp\\faces\\"
img = io.imread(path + "1.jpg")



for i in range(1,7):
    img = io.imread(path + f"{i}.jpg")
    raw=cv2.imread(path + f"{i}.jpg")
    # 特征提取器的实例化
    dets = detector(img)
    print("人脸数：", len(dets))
    # 输出人脸矩形的四个坐标点
    for j, d in enumerate(dets):
        print("第", j, "个人脸d的坐标：",
              "left:", d.left(),
              "right:", d.right(),
              "top:", d.top(),
              "bottom:", d.bottom())
        out=raw[d.top()-1:d.bottom(),d.left()-1:d.right()]
        cv2.imwrite(path + f"{i}_face.jpg",out)


# # 绘制图片
# win = dlib.image_window()
# # 清除覆盖
# # win.clear_overlay()
# win.set_image(img)
# # 将生成的矩阵覆盖上
# win.add_overlay(dets)
# # 保持图像
# dlib.hit_enter_to_continue()