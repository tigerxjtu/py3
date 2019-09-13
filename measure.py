import cv2
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from cut import shoulder_points
import math

# path1=r'C:\dataguru_new\txtdata'
# path2=r'C:\projects\python\measure\ui\data\pics'

path1=r'C:\projects\python\measure\ui\data\txtdata'
path2=r'C:\projects\python\measure\ui\data\pics'



#获取文件名的ID含义部分
def get_name(file_name):
    return file_name[:-6]

#遍历文件得到所有id列表
def list_name(path):
    names={}
    for file_name in os.listdir(path):
        names[get_name(file_name)]=1
    return [key for key in names.keys()]

#获取所有图片文件id、文件路径 pair
def list_pic_files(path):
    file_names={}
    for root,dirs,files in os.walk(path):
        for name in files:
            file_names[name]=os.path.join(root,name)
    return file_names

#获取特征文件对应的轮廓点或特征点
def get_points(file):
    with open(file) as f:
        content = f.read()
        data = json.loads(content)
        points = data['featureXY']
        return [(int(p['x']), int(p['y'])) for p in points]

names=list_name(path1)
pics=list_pic_files(path2)

#根据文件id和标签得到对应的特征文件和图片文件路径
def get_file_name(name,tag):
    file_name='%s%s1.txt'%(name,tag)
    pic_name='%s%s.jpg'%(name,tag)
    print(file_name,pic_name,pics[pic_name])
    return os.path.join(path1,file_name),pics[pic_name]

#shoulder: 0.17~0.22
def shoulder_bound(max_y,min_y):
    lower, upper = 0.17, 0.22
    return part_bound(max_y, min_y, lower, upper)

def neck_bound(max_y,min_y):
    lower, upper = 0.1, 0.17
    return part_bound(max_y,min_y,lower,upper)

def part_bound(max_y,min_y,lower,upper):
    h = max_y - min_y
    lower_y = min_y + lower * h
    upper_y = min_y + upper * h
    return int(lower_y), int(upper_y)

def percent_y(max_y,min_y,percent):
    h = max_y - min_y
    y = min_y + percent * h
    return int(y)

def calculate_height(points):
    X = np.array([p[0] for p in points])
    Y = np.array([p[1] for p in points])
    max_y = np.max(Y)
    min_y = np.min(Y)
    return int(max_y-min_y)

def is_neighbor(p1,p2):
    x1, y1 = p1
    x2, y2 = p2
    dis=abs(x1-x2)+abs(y1-y2)
    return dis<=20

def distance(p1,p2):
    x1,y1=p1
    x2,y2=p2
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def angle(center,left,right):
    a=distance(left,center)
    b=distance(center,right)
    c=distance(left,right)
    ang=c/(a+b)
    return ang

def angle_point(points,delta=5):
    size=len(points)
    angles=[1 for _ in range(size)]
    for i in range(delta+20,size-delta):
        angles[i]=angle(points[i],points[i-delta],points[i+delta])
    index=np.argmin(angles)
    return index,points[index]

def shoulder_markers(points):
    X = np.array([p[0] for p in points])
    Y = np.array([p[1] for p in points])
    max_y = np.max(Y)
    min_y = np.min(Y)
    lower_y, upper_y = shoulder_bound(max_y, min_y)

    shoulder = np.where(Y >= lower_y)
    Y = Y[shoulder]
    X = X[shoulder]

    shoulder = np.where(Y <= upper_y)
    Y = Y[shoulder]
    X = X[shoulder]

    points=zip(X,Y)
    part1=[]
    part2=[]
    for p in points:
        if part1:
            last=part1[-1]
            if is_neighbor(last,p):
                part1.append(p)
            else:
                if part2:
                    if not is_neighbor(part2[-1],p):
                        print(p,'is invalid', last)
                        continue
                part2.append(p)
        else:
            part1.append(p)
    print(part1)
    print(part2)
    _,point1=angle_point(part1)
    _,point2=angle_point(part2)
    print(point1,point2)
    return point1,point2


def calculate_feature(points):
    X=np.array([p[0] for p in points])
    Y=np.array([p[1] for p in points])
    max_y=np.max(Y)
    min_y=np.min(Y)
    lower_y,upper_y=neck_bound(max_y,min_y)

    neck=np.where(Y>=lower_y)
    Y=Y[neck]
    X=X[neck]

    neck = np.where(Y <= upper_y)
    Y = Y[neck]
    X = X[neck]

    # (x1,x2,y,distance)
    result=np.zeros((upper_y-lower_y+1,3),dtype=np.int)
    distances=np.zeros((upper_y-lower_y+1),dtype=np.int)
    i=0
    for y in range(lower_y,upper_y+1):
        result[i,2]=y
        index=np.where(Y==y)
        x=X[index]
        result[i,0]=np.min(x)
        result[i,1]=np.max(x)
        distances[i]=result[i,1]-result[i,0]
        i+=1
    index=np.where(distances>0)
    result=result[index]
    distances=distances[index]
    ind=np.argmin(distances)

    return max_y,min_y,(result[ind,0],result[ind,1],result[ind,2])

# feature_file,_=get_file_name(names[0],'F')
# points=get_points(feature_file)
# print(calculate_feature(points))

def front_features(name):
    feature_file, _ = get_file_name(name, 'F')
    points = get_points(feature_file)
    max_y,min_y,neck = calculate_feature(points)
    return int(max_y-min_y),int(neck[1]-neck[0])

def side_features(name):
    feature_file, _ = get_file_name(name, 'S')
    points = get_points(feature_file)
    max_y,min_y,neck = calculate_feature(points)
    return int(max_y-min_y),int(neck[1]-neck[0])

def back_features(name):
    feature_file, _ = get_file_name(name, 'B')
    points = get_points(feature_file)
    max_y,min_y,neck = calculate_feature(points)
    return int(max_y-min_y),int(neck[1]-neck[0])

#shoulder: 0.17~0.22
#xiong： 0.23~0.29
#yao： 0.36~0.42
#展示正面图片和轮廓或特征点
def display_front(name,tag='F'):
    feature,img=get_file_name(name,tag)
    f_img=cv2.imread(img)
    f_points=get_points(feature)
    print(f_points)
    max_y,min_y,neck=calculate_feature(f_points)
    lower_y, upper_y=neck_bound(max_y,min_y)
    plt.figure()
    plt.imshow(f_img[:, :, ::-1])
    plt.scatter([p[0] for p in f_points], [p[1] for p in f_points], marker='+', color='r', s=1)
    percents=[0.17,0.22,0.23,0.29,0.36,0.42]
    for percent in percents:
        plt.scatter(range(750),[percent_y(max_y,min_y,percent)]*750,marker='.', color='g', s=1)
    p1,p2=shoulder_markers(f_points)

    x=[p1[0],p2[0]]
    y=[p1[1],p2[1]]
    plt.scatter(x, y, marker='*', color='b', s=10)

    plt.show()

#一起展示正面、侧面、背面图片和轮廓点
def display(name):
    feature,img=get_file_name(name,'F')
    f_img=cv2.imread(img)
    f_points=get_points(feature)

    feature, img = get_file_name(name, 'S')
    s_img = cv2.imread(img)
    s_points = get_points(feature)

    feature, img = get_file_name(name, 'B')
    b_img = cv2.imread(img)
    b_points = get_points(feature)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('Front')
    plt.imshow(f_img[:, :, ::-1])
    plt.scatter([p[0] for p in f_points],[p[1] for p in f_points],marker = '+', color='r',s=1)
    plt.subplot(2, 2, 2)
    plt.title('Side')
    plt.imshow(s_img[:, :, ::-1])
    plt.scatter([p[0] for p in s_points], [p[1] for p in s_points], marker='+', color='r',s=1)
    plt.subplot(2, 2, 3)
    plt.title('Back')
    plt.imshow(b_img[:, :, ::-1])
    plt.scatter([p[0] for p in b_points], [p[1] for p in b_points], marker='+', color='r',s=1)
    plt.show()


path3=r'C:\data\measure'


userdata=json.load(open(os.path.join(path3,'userdata.json'),encoding='UTF-8'))
bodydata=json.load(open(os.path.join(path3,'bodydata.json'),encoding='UTF-8'))


def find_user(bd):
    record = {}
    for ud in userdata:
        if bd['name']==ud['姓名']:
            record['name']=bd['name']
            record['height']=bd['height']
            record['weight']=bd['weight']
            record['neck']=ud['领围']
            record['shoulder']=ud['肩宽']
            return record
    return None

# for ud in userdata:
#     pic_path=ud['pic']
#     file_name=pic_path.split()
# bd=bodydata[0]

def join_userdata():
    records=[]
    for bd in bodydata:
        pic_path=bd['pic']
        file_name=pic_path.split('/')[-1]
        # print(file_name)
        name=file_name[:-5]
        # print(name)
        if name in names:
            # print(name)
            record=find_user(bd)
            if record:
                try:
                    record['feature']=name
                    fh,fw=front_features(name)
                    sh,sw=side_features(name)
                    bh,bw=back_features(name)
                    record['fh']=fh
                    record['fw']=fw
                    record['sh']=sh
                    record['sw']=sw
                    record['bh']=bh
                    record['bw']=bw
                    # print(record)
                    records.append(record)
                except Exception as e:
                    print(e)
    # print(names)
    print(records)
    print(len(records))
    json.dump(records,open(os.path.join(path3,'records.json'),'w',encoding='gbk'))
    records=json.load(open(os.path.join(path3,'records.json')))
    print(records[0])

def shoulder_feature(name):
    feature_file, pic_file = get_file_name(name, 'F')
    left,right = shoulder_points(pic_file)
    points = get_points(feature_file)
    return int(left[0]- right[0]), int(left[1] - right[1]), calculate_height(points) #dx,dy,h

def join_user_shoulder_data():
    records = []
    for bd in bodydata:
        pic_path = bd['pic']
        file_name = pic_path.split('/')[-1]
        # print(file_name)
        name = file_name[:-5]
        # print(name)
        if name in names:
            # print(name)
            record = find_user(bd)
            if record:
                try:
                    record['feature'] = name
                    dx,dy,fh=shoulder_feature(name)
                    record['fh'] = fh
                    record['dx'] = dx
                    record['dy'] = dy
                    # print(record)
                    records.append(record)
                except Exception as e:
                    print(e)
    # print(names)
    print(records)
    print(len(records))
    json.dump(records, open(os.path.join(path3, 'shoulder_records.json'), 'w', encoding='gbk'))
    records = json.load(open(os.path.join(path3, 'shoulder_records.json')))
    print(records[0])

def display_body(name,tag='F'):
    feature, img = get_file_name(name, tag)
    points = get_points(feature)
    img=cv2.imread(img)
    height,width = img.shape[:2]
    for point in points:
        cv2.circle(img,point,1,(0,0,255))
    img=cv2.resize(img,(width//2,height//2))
    cv2.imshow("img",img)
    cv2.waitKey()

if __name__ == '__main__':
    # f_img=cv2.imread(r'C:/dataguru_new/pics/201810/U1000154181021201534789F.jpg')
    # print(f_img.shape)

    # display(names[0])
    # display_front(names[0])
    # display('U1000273181024150149259')
    # display_front('U1000240181024143003419', "F")

    # print(get_name('U1000154181021201534789B1.txt'))
    # print(list_name(path1))

    # join_userdata()
    # join_user_shoulder_data()

    # U1000208181024132310246

    # display_front('U1002217190901092403591', "F")
    # display_front('U1000208181024132310246', "B")

    display_body('U1002217190901092403591','F')