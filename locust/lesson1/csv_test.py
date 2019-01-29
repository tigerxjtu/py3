#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: csv_test.py
@time: 2019-01-09
'''

import csv

csvfile = open('csvWrite.csv', 'w', newline='')
writer = csv.writer(csvfile)
writer.writerow(('编号', '网址', '关键字'))
ss = [
    ('1', 'http://nnzhp.cn/', '牛牛'),
    ('2', 'http://www.baidu.com/', '百度'),
    ('3', 'http://www.jd.com/', '京东')
]
ccs = ('4', 'http://http://www.cnblogs.com/hhfzj/', '自己博客')
writer.writerows(ss)
writer.writerow(ccs)
csvfile.close()

csvfile = open('csvWrite.csv', newline='')  # 打开一个文件
csvReader = csv.reader(csvfile)  # 返回的可迭代类型
print(type(csvReader))
for content in csvReader:
    print(content)
csvfile.close()  # 关闭文件
# 读取csv文件方法2
# import csv
with open('csvWrite.csv', newline='') as csvfile:  # 此方法:当文件不用时会自动关闭文件
    csvReader = csv.reader(csvfile)
    for content in csvReader:
        print(content)



