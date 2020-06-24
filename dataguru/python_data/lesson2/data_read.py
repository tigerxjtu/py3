import csv
import os

path = r"C:\dataguru_new\Python数据处理实战：基于真实场景的数据\第二课\数据和代码"
data = []
with open(os.path.join(path, 'contest_ext_crd_is_creditcue.csv')) as f:
    csvreader = csv.reader(f, delimiter=',')
    header = csvreader.__next__()
    for row in csvreader:
        data.append(row)
print(header)
print(data)

with open(os.join(path, 'out.csv'), 'w') as f2:
    cw = csv.writer(f2, lineterminator='\n')
    cw.writerow(header)
    for row in data:
        cw.writerow(row)
    # cw.writerows(data)

import json

with open(os.path.join(path, 'sample1.json'), encoding='utf-8') as f:
    f_read = f.read()
json.loads(f_read)
data = data['data']
print(data)

with open(os.path.join(path, 'json格式数据.json'), 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)  # 确保中文正常显示


