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


def compute(true_rect,detect_rect):
    intersect_rect = rect_of_intersect(true_rect,detect_rect)
    print(intersect_rect,true_rect,detect_rect)
    if not intersect_rect:
        print("not intersected")
        return (0,0,0)

    intersect_area=area_of_rect(intersect_rect)
    true_area=area_of_rect(true_rect)
    detect_area=area_of_rect(detect_rect)
    print(intersect_area,true_area,detect_area)
    precision=intersect_area/detect_area
    recall=intersect_area/true_area
    iou=intersect_area/(true_area+detect_area-intersect_area)
    return (precision,recall,iou)


if __name__ == '__main__':
    true_rect=convert_rect(0.4,0.7,0.2,0.2)
    detect_rect=convert_rect(0.35,0.55,0.5,0.1)
    (precision, recall, iou) = compute(true_rect,detect_rect)
    print("precision={}, recall={}, iou={}".format(precision, recall, iou))



