import numpy as np
import cv2

def area_of_rect(rect):
    (left_x, left_y, right_x, right_y)=rect
    w,h=right_x-left_x,right_y-left_y
    return w*h

def rect_of_intersect(rect1,rect2):
    (x11, y11, x12, y12) = rect1  # 矩形1左上角(x11,y11)和右下角(x12,y12)
    (x21, y21, x22, y22) = rect2  # 矩形2左上角(x21,y21)和右下角(x22,y22)
    # 我们在此假想一个大矩形，它正好把坐标轴区域上矩形1和矩形2包围起来，我们称它为外包框
    StartX = min(x11, x21)  # 外包框在x轴上左边界
    EndX = max(x12, x22)  # 外包框在x轴上右边界
    StartY = min(y11, y21)  # 外包框在y轴上上边界
    EndY = max(y12, y22)  # 外包框在y轴上下边界
    # 相交矩形区域的宽度和高度，因为这两个矩形的边都是平行与坐标轴的，因此他们的相交区域也是矩形
    # 那什么条件下才能形成相交区域啦，矩形是二维的，只有x,y方向都有交集时，他们才能形成相交区域
    # 记住上面提到过得外包框，就比较容易理解相交的条件了,blog1中的逆向思维也是不错的，不过感觉还是有些麻烦不直观，
    # 因此在这里多解释一下
    CurWidth = (x12 - x11) + (x22 - x21) - (EndX - StartX)  # (EndX-StartX)表示外包框的宽度
    CurHeight = (y12 - y11) + (y22 - y21) - (EndY - StartY)  # (Endy-Starty)表示外包框的高度

    if CurWidth <= 0 or CurHeight <= 0:  # 不相交
        return None
    else:  # 相交
        X1 = max(x11, x21)  # 有相交则相交区域位置为：小中取大为左上角，大中取小为右下角
        Y1 = max(y11, y21)
        X2 = min(x12, x22)
        Y2 = min(y12, y22)
        IntersectRect = (X1, Y1, X2, Y2)
        return IntersectRect

def compute_iou(true_rect,detect_rect):
    intersect_rect = rect_of_intersect(true_rect,detect_rect)
    if not intersect_rect:
        # print("not intersected")
        return 0

    intersect_area=area_of_rect(intersect_rect)
    true_area=area_of_rect(true_rect)
    detect_area=area_of_rect(detect_rect)
    # print(intersect_area,true_area,detect_area)
    iou=intersect_area/(true_area+detect_area-intersect_area)
    return iou


def calc_iou(idx1,idx2,data):
    iou=compute_iou(data[idx1][1:],data[idx2][1:])
    print(iou)
    return iou

def process_nms(data,idx_candidates,iou_thresh):
    keep=[]
    while len(idx_candidates)>0:
        idx = idx_candidates[0]
        keep.append(idx)
        idx_candidates=[i for i in idx_candidates[1:] if calc_iou(idx,i,data) < iou_thresh]
        # print(idx_candidates)
    return keep

def compute_nms(data,iou_thresh):
    scores=data[:,0]
    candidates = scores.argsort()[::-1]
    return process_nms(data,candidates,iou_thresh)


if __name__ == '__main__':
    data = np.array([[0.8,133,199,203,341],[0.6,125,195,208,346],[0.7,130,225,205,320]])
    print(data)
    result = compute_nms(data,0.3)
    print(result)
    img = cv2.imread('horse_dog_car_person.jpg')
    colors=[(0,0,255),(0,255,0),(255,0,0)]# red, green, blue
    img_dst = img.copy()

    for i in range(data.shape[0]):
        pt1=(int(data[i][1]),int(data[i][2]))
        pt2=(int(data[i][3]),int(data[i][4]))
        # print(pt1,pt2)
        cv2.rectangle(img,pt1,pt2,colors[i],2)
    cv2.imshow('orig',img)

    for i in result:
        pt1 = (int(data[i][1]), int(data[i][2]))
        pt2 = (int(data[i][3]), int(data[i][4]))
        print(pt1, pt2)
        cv2.rectangle(img_dst, pt1, pt2, colors[i], 2)
    cv2.imshow('dst', img_dst)
    cv2.waitKey(0)
