# Q1 本章第3节视频，关于指标计算， 请计算一下每个品种(规格)的烟的订购次数, 结果以字典形式展示，
# 并对字典键对应的值进行降序排序，结果如下:
import json
import os

path = r"C:\dataguru_new\Python数据处理实战：基于真实场景的数据\第二课\数据和代码"
with open(os.path.join(path, 'sample3.json'), encoding='utf-8') as f:
    f_read = f.read()
data = json.loads(f_read)
data = data['data']
print(len(data['indent']))
order_tradeName = []
for i in data['indent']:
    for j in i['indentDetails']:
        if float(j.get('orderQuantity', 0)) > 0:
            order_tradeName.append((i.get('indentNum', 0), j.get('tradeName', '')))
order_tradeName = [(a, b) for a, b in order_tradeName if a != 0 and b != '']

order_times = {}
for order, tradeName in order_tradeName:
    order_times[tradeName] = order_times.get(tradeName, 0) + 1
order_times_descend = sorted(order_times.items(), key=lambda item: item[1], reverse=True)
result = {k: v for k, v in order_times_descend}
for k, v in result.items():
    print(k, v)

# Q2 统计一下订购次数排名前100的烟的品种以及每个品种的订购次数。
result = order_times_descend[:100]
for a, b in result:
    print(a, b)

# Q3 对于sample2.json数据, 统计一下有多少个IC类指标(在content键里面, IC指标是指以IC开头的指标)
with open(os.path.join(path, 'sample2.json'), encoding='utf-8') as f:
    f_read = f.read()
data = json.loads(f_read)
diff_types = [i for i in data['content']]
diff_types = [i for i in diff_types if i[:2] == 'IC']
print(diff_types[:10])
print(len(set(diff_types)))

# Q4 对于sample1.json数据, 统计一下每个大类指标的个数, 比如EG类指标有多少，IG类指标有多少个, 展示结果如下(字典的键-值不分顺序):
with open(os.path.join(path, 'sample1.json'), encoding='utf-8') as f:
    f_read = f.read()
data = json.loads(f_read)
diff_types = {}
for key in data['content']:
    k = key.split('_')[0]
    diff_types[k] = diff_types.get(k, 0) + 1
print(diff_types)
for a, b in diff_types.items():
    print(a, b)
