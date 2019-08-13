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


def bodypart(filename):
    APP_ID = '44eab39f0ed44844b94f487a6e88fdbc'  # 'd90755ad2f2047dbabb12ad0adaa0b03'
    API_KEY = '55e735d6888b46908915f3533a6b7442'  # '22f1025b88f54494bcf5d980697b4b83 '
    SECRET_KEY = '41213cbdaffa483d9ed9a59a24157d4b'  # '4a4b41139c204905be1db659d751355f'

    #    APP_ID = 'd90755ad2f2047dbabb12ad0adaa0b03'
    #    API_KEY = 'b7b0f4d01f7f4aef9b5453f6558c23b1'
    #    SECRET_KEY = '6ad666162ef24213b5bde7bdd904fcbe'

    client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)

    # """ 读取图片 """
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    image = get_file_content(filename)

    # """ 调用人体关键点识别 """
    para = client.bodyAnalysis(image)
    time.sleep(2)
    return para





def body_points(file_path):
    para = bodypart(file_path)
    # print(para)
    person=para['person_info'][0]
    parts=person['body_parts']
    # for key in parts:
    #     print(key,parts[key])
    dir(parts)
    points=[(k,(v['x'],v['y'])) for k,v in parts.items() ]
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


if __name__ == '__main__':
    # U1000243181024143406116
    name='U1000208181024132310246'
    file_path=find_name(name)
    print(file_path)
    # file_path = r'C:\dataguru_new\pics\201810\U1000284181024155802594F.jpg'
    # left_shoulder,right_shoulder
    points=body_points(file_path)
    f_img=cv2.imread(file_path)

    plt.figure()
    plt.imshow(f_img[:, :, ::-1])
    plt.scatter([p[1][0] for p in points], [p[1][1] for p in points], marker='+', color='r', s=1)
    for p in points:
        plt.annotate(p[0],p[1])
    plt.show()



