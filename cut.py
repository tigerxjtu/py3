import requests
import numpy as np
import skimage.morphology as sm
from skimage import measure
import cv2
from aip import AipBodyAnalysis
import urllib3
import base64
import json
from urllib.parse import urlencode
import matplotlib.pyplot as plt
import time
import os

APP_ID = '44eab39f0ed44844b94f487a6e88fdbc'  # 'd90755ad2f2047dbabb12ad0adaa0b03'
API_KEY = '55e735d6888b46908915f3533a6b7442'  # '22f1025b88f54494bcf5d980697b4b83 '
SECRET_KEY = '41213cbdaffa483d9ed9a59a24157d4b'  # '4a4b41139c204905be1db659d751355f'

threshfeng=5#填补缝隙的结构大小

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def bodyseg(filename):

    #    APP_ID = 'd90755ad2f2047dbabb12ad0adaa0b03'
    #    API_KEY = 'b7b0f4d01f7f4aef9b5453f6558c23b1'
    #    SECRET_KEY = '6ad666162ef24213b5bde7bdd904fcbe'

    client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)

    # """ 读取图片 """
    image = get_file_content(filename)

    res = client.bodySeg(image)
    labelmap = base64.b64decode(res['labelmap'])
    # time.sleep(2)
    nparr_labelmap = np.fromstring(labelmap, np.uint8)
    labelmapimg = cv2.imdecode(nparr_labelmap, 1)
    print(labelmapimg.shape)
    im_new_labelmapimg = np.where(labelmapimg == 1, 255, labelmapimg)
    # print(im_new_labelmapimg.shape)
    # img=cv2.cvtColor(im_new_labelmapimg, cv2.COLOR_BGR2GRAY)
    # return img.astype('uint8')
    # cv2.imwrite('outline.png',im_new_labelmapimg)
    return im_new_labelmapimg


def bodypart(filename):

    #    APP_ID = 'd90755ad2f2047dbabb12ad0adaa0b03'
    #    API_KEY = 'b7b0f4d01f7f4aef9b5453f6558c23b1'
    #    SECRET_KEY = '6ad666162ef24213b5bde7bdd904fcbe'

    client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)

    # """ 读取图片 """
    image = get_file_content(filename)

    # """ 调用人体关键点识别 """
    para = client.bodyAnalysis(image)
    # time.sleep(2)
    return para

def get_outline(img,rect):
    x,y,w,h=rect
    body=img[y:y+h,x:x+w]
    # cv2.imwrite('outline.png',body)
    # print(body.shape)
    edges = cv2.Canny(body, 10, 100)
    edgesmat = np.mat(edges)
    # points=[]
    # for i in range(h):
    #     Rowdata = edgesmat[i, :].tolist()[0]
    #     for j in range(len(Rowdata)):
    #         if Rowdata[j] == 255:
    #             points.append((j+x,i+y))
    # print(edgesmat.shape)
    points = [(j+x,i+y) for i in range(h) for j in range(w) if edgesmat[i,j] == 255]
    # print(points)
    return points

def rect(x,y,w,h,max_w,max_h,ratio=0.05):
    left = int(max(x - ratio * w, 0))
    top = int(max(y-ratio*h,0))
    width = int(min((1+2*ratio)*w,max_w-left))
    height = int(min((1+2*ratio)*h,max_h-top))
    return left,top,width,height

def display_body(file_path):
    para = bodypart(file_path)
    persons = para['person_info']
    parts = persons[0]['body_parts']
    print(parts)
    img=cv2.imread(file_path)
    height,width = img.shape[:2]
    orig = img.copy()
    # points = [(int(v['x']), int(v['y'])) for k, v in parts.items()]
    points=[]
    img_seg = bodyseg(file_path)
    for person in persons:
        loc=person['location']
        x_left=int(loc['left'])
        y_top = int(loc['top'])
        w=int(loc['width'])
        h=int(loc['height'])
        x_left,y_top,w,h=rect(x_left,y_top,w,h,width,height)
        cv2.rectangle(img,(x_left,y_top),(x_left+w,y_top+h),(0,255,0),2)
        if not points:
            points = get_outline(img_seg, (x_left,y_top,w,h))
            # body_img=orig[y_top:y_top+h,x_left:x_left+w,:]
            # cv2.imwrite('body.png',body_img)
    for point in points:
        cv2.circle(img,point,1,(0,0,255))

    cv2.imshow("img",img)
    cv2.waitKey()

# def display_img(img):

def body_points(file_path):
    para = bodypart(file_path)
    # print(para)
    person=para['person_info'][0]
    parts=person['body_parts']
    # for key in parts:
    #     print(key,parts[key])
    # dir(parts)
    points={ k:(v['x'],v['y']) for k,v in parts.items() }
    # points=[ (k,v) for k,v in parts.items ]
    return points

def shoulder_points(file_path):
    para = bodypart(file_path)
    person = para['person_info'][0]
    parts = person['body_parts']
    left_shoulder=parts['left_shoulder']
    right_shoulder=parts['right_shoulder']
    return (left_shoulder['x'],left_shoulder['y']),(right_shoulder['x'],right_shoulder['y'])

def find_name(name):
    base_path=r'C:\dataguru_new\pics'
    dirs=os.listdir(base_path)
    for d in dirs:
        path=os.path.join(base_path,d)
        file=os.path.join(path,name+'F.jpg')
        # print(file)
        if os.path.exists(file):
            return file
    print(file+' not exists')
    return None

def get_pics(name):
    base_path = r'C:\dataguru_new\pics'
    dirs = os.listdir(base_path)
    path=''
    for d in dirs:
        path = os.path.join(base_path, d)
        file = os.path.join(path, name + 'F.jpg')
        # print(file)
        if os.path.exists(file):
            break
    else:
        return None
    return [os.path.join(path, name + 'F.jpg'),os.path.join(path, name + 'B.jpg'),os.path.join(path, name + 'S.jpg')]

if __name__ == '__main__':
    # U1000243181024143406116
    name='U1000208181024132310246'
    file_path=find_name(name)
    print(file_path)

    file_path = r'C:\dataguru_new\pics\201810\U1000208181024132310246F.jpg'
    # left_shoulder,right_shoulder
    # left_ankle
    points=body_points(file_path)
    # str=json.dumps(points)
    # f=open('points.json','w')
    # f.write(str)
    # f.close()
    f_img=cv2.imread(file_path)

    plt.figure()
    plt.imshow(f_img[:, :, ::-1])
    plt.scatter([p[0] for p in points.values()], [p[1] for p in points.values()], marker='+', color='r', s=1)
    for k,p in points.items():
        plt.annotate(k,p)
    plt.show()

    # files=get_pics(name)
    # for file in files:
    #     display_body(file)
    #     break
    # display_body('201005100004_Front.jpg')

    # img=bodyseg(file_path)
    # cv2.imshow("seg",img)
    # cv2.waitKey()

    # filename=r'C:\dataguru_new\pics\201810\U1000208181024132310246F.jpg'
    # APP_ID = '44eab39f0ed44844b94f487a6e88fdbc'  # 'd90755ad2f2047dbabb12ad0adaa0b03'
    # API_KEY = '55e735d6888b46908915f3533a6b7442'  # '22f1025b88f54494bcf5d980697b4b83 '
    # SECRET_KEY = '41213cbdaffa483d9ed9a59a24157d4b'  # '4a4b41139c204905be1db659d751355f'
    #
    # #    APP_ID = 'd90755ad2f2047dbabb12ad0adaa0b03'
    # #    API_KEY = 'b7b0f4d01f7f4aef9b5453f6558c23b1'
    # #    SECRET_KEY = '6ad666162ef24213b5bde7bdd904fcbe'
    #
    # client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)
    #
    # # """ 读取图片 """
    # image = get_file_content(filename)
    #
    # # """ 调用人体关键点识别 """
    # res = client.bodySeg(image)
    #
    # foreground = base64.b64decode(res['foreground'])
    # labelmap = base64.b64decode(res['labelmap'])
    # scoremap = base64.b64decode(res['scoremap'])
    #
    # nparr_foreground = np.fromstring(foreground, np.uint8)
    # foregroundimg = cv2.imdecode(nparr_foreground, 1)
    # foregroundimg = cv2.resize(foregroundimg, (512, 512), interpolation=cv2.INTER_NEAREST)
    # im_new_foreground = np.where(foregroundimg == 1, 10, foregroundimg)
    # cv2.imwrite('foreground.png', im_new_foreground)
    #
    # nparr_labelmap = np.fromstring(labelmap, np.uint8)
    # labelmapimg = cv2.imdecode(nparr_labelmap, 1)
    # labelmapimg = cv2.resize(labelmapimg, (512, 512), interpolation=cv2.INTER_NEAREST)
    # im_new_labelmapimg = np.where(labelmapimg == 1, 255, labelmapimg)
    # cv2.imwrite('labelmap.png', im_new_labelmapimg)
    #
    # nparr_scoremap = np.fromstring(scoremap, np.uint8)
    # scoremapimg = cv2.imdecode(nparr_scoremap, 1)
    # scoremapimg = cv2.resize(scoremapimg, (512, 512), interpolation=cv2.INTER_NEAREST)
    # im_new_scoremapimg = np.where(scoremapimg == 1, 255, scoremapimg)
    # cv2.imwrite('scoremap.png', im_new_scoremapimg)


