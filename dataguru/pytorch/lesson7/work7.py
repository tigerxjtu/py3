import cv2
import sys
import os
import json
import base64
import datetime
import insightface
import numpy as np
from io import BytesIO
from gevent.pywsgi import WSGIServer
from flask import Flask, request, jsonify, abort


class FaceAnalysis:
    def __init__(self, gpu_num=-1, nms=0.4):
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id=gpu_num, nms=nms)

    def __call__(self, x):
        """
        :param x: RGB [m, n, 3] for cv2.imread, not RGB.
        :return:
        """
        faces = self.model.get(x)
        for idx, face in enumerate(faces):
            print("Face [%d]:" % idx)
            print("\tage:%d" % (face.age))
            gender = 'Male'
            if face.gender == 0:
                gender = 'Female'
            print("\tgender:%s" % (gender))
            print("\tembedding shape:%s" % face.embedding.shape)
            print("\tbbox:%s" % (face.bbox.astype(np.int).flatten()))
            print("\tlandmark:%s" % (face.landmark.astype(np.int).flatten()))
            print("")
        assert len(faces) == 1, "This image must only have one face "

        return {'age': face.age, 'gender': gender, 'feature_vector': face.embedding.tolist()}


app = Flask(__name__)
# 常量
# blackmanba = FaceAnalysis(int(sys.argv[1]))
blackmanba = FaceAnalysis()

faces={}

def nearest_face(feature):
    cur_name=''
    min_dis=None
    for name,vector in faces.items():
        dis=distance(feature,vector)
        if not min_dis:
            cur_name=name
            min_dis=dis
        else:
            if dis<min_dis:
                cur_name = name
                min_dis = dis
    return cur_name,min_dis

def distance(v1,v2):
    diff=[abs(float(i1)-float(i2)) for i1,i2 in zip(v1,v2)]
    return np.sum(diff)

path=r'C:\tmp\faces'
face_files=['1_face.jpg','3_face.jpg','6_face.jpg']
names=['张三','李四','王五']

for name,file in zip(names,face_files):
    face_path=os.path.join(path,file)
    img=cv2.imread(face_path)
    img=cv2.resize(img,(112,112))
    feature=blackmanba(img[...,::-1])
    faces[name]=feature['feature_vector']
print(faces)

@app.route('/faceanalysis', methods=['POST'])
def requests():
    data = request.get_json(force=True)
    if (not data or not 'reqid' in data) or (not 'reqtime' in data) or (not 'image' in data) > 1:
        abort(400)

    pic = BytesIO()
    pic.write(base64.b64decode(bytes(data["image"], encoding='utf8')))
    pic.seek(0)
    file_bytes = np.asarray(bytearray(pic.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)[..., ::-1]

    raw_result = blackmanba(img)

    result = {
        'retid': data['reqid'],
        'rettime': datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
    }
    result.update(raw_result)
    print(raw_result)
    print(raw_result['feature_vector'])
    name,dis=nearest_face(raw_result['feature_vector'])
    result['name']=name
    result['distance']=dis
    result = json.dumps(result)
    return jsonify(result)


if __name__ == '__main__':
    ip = '0.0.0.0'
    post = 5000
    print('running server http://{0}'.format(ip + ':' + str(post)))
    WSGIServer((ip, post), app).serve_forever()
