import base64
import datetime
import requests
import datetime
import json

file = r'C:\tmp\faces\2_face.jpg'
# with open('Tom_Hanks_54745.png', 'rb') as f:
with open(file, 'rb') as f:
    images_data = base64.b64encode(f.read())

data = {
    "reqid":"234567890",
    "reqtime": datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
    "image" : str(images_data, encoding='utf-8')
}

result=requests.post("http://127.0.0.1:5000/faceanalysis", json=data).json()
result=json.loads(result)
print(result['name'])