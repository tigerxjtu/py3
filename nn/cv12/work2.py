import numpy as np

def convert_rect(cent_x,cent_y,w,h):
    left_x,left_y=cent_x-w/2,cent_y-h/2
    right_x,right_y=cent_x+w/2,cent_y+h/2
    return (left_x,left_y,right_x,right_y)


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

def pos_index(index,shape):
    sw,sh=shape[0],shape[1]
    return int(index/sh),index%sh


def calc_iou(idx1,idx2,shape,data):
    ix1,iy1=pos_index(idx1,shape)
    ix2,iy2=pos_index(idx2,shape)
    x1,x2=data[idx1,1]+ix1,data[idx2,1]+ix2
    y1,y2=data[idx1,2]+iy1,data[idx2,2]+iy2
    w1,w2=data[idx1,3],data[idx2,3]
    h1, h2 = data[idx1, 4], data[idx2, 4]
    iou=compute_iou(convert_rect(x1,y1,w1,h1),convert_rect(x2,y2,w2,h2))
    # print(iou)
    return iou


def process_nms(data,idx_candidates,shape,iou_thresh):
    keep=[]
    while len(idx_candidates)>0:
        idx = idx_candidates[0]
        keep.append(idx)
        idx_candidates=[i for i in idx_candidates[1:] if calc_iou(idx,i,shape,data) < iou_thresh]
        # print(idx_candidates)
    return keep


def compute_nms(data,shape,pc_thresh,iou_thresh):
    scores=data[:,0]
    order = scores.argsort()[::-1]
    candidates=[]
    for index in order:
        if scores[index]<pc_thresh:
            break
        candidates.append(index)
    return process_nms(data,candidates,shape,iou_thresh)


data = [0.22,1,0.8,2.2,0.4,
        0.45,0.7,0.1,3.1,2.4,
        0.83,0.3,0.5,2.7,0.2,
        0.29,0.9,1,1.6,0.7,
        0.49,0.3,0.7,2.8,0.7,
        0.11,0.4,0.1,3.4,1.8,
        0.05,0.4,0,1.6,0.9,
        0.70,0.1,0,2.4,1.1,
        0.78,0.9,0.1,0.4,1.8,
        0.83,0.3,0.8,1,0.9,
        0.82,0.8,0.1,2.4,2.5,
        0.61,0.8,0.8,1.5,0.6,
        0.72,0.9,0,1,1.6,
        0.03,0.8,0.5,2.5,1.5,
        0.65,0,0.1,2.1,2,
        0.12,0.5,0.1,1.2,2.5,
        0.27,0.7,0.2,3.5,1.1,
        0.72,0.7,0,1.4,0.3,
        0.10,0.8,0.6,2.2,1.3,
        0.35,1,0,3,2.5,
        0.17,0.8,0.9,2.3,1.3,
        0.01,0,0.9,2.5,2,
        0.74,0.5,0.5,2.8,0.1,
        0.58,0.5,1,2.7,0.3,
        0.75,0.5,0.6,0.6,0.5]

data=np.array(data,dtype=np.float)
data=data.reshape(-1,5)
keep=compute_nms(data,(5,5),0.4,0.5)
print('nms result:',len(keep))
for index in keep:
    print('index:{},p:{},x:{},y:{},w:{},h:{}'.format(pos_index(index,(5,5)),data[index,0],data[index,1],data[index,2],data[index,3],data[index,4]))