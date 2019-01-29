#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: create_csv.py
@time: 2019-01-21
'''

import csv

csvfile = open('data.csv', 'w', newline='')
writer = csv.writer(csvfile)
writer.writerow(('key', 'phone', 'passwd'))
# ss = [
#     ('1', 'http://nnzhp.cn/', '牛牛'),
#     ('2', 'http://www.baidu.com/', '百度'),
#     ('3', 'http://www.jd.com/', '京东')
# ]
# ccs = ('4', 'http://http://www.cnblogs.com/hhfzj/', '自己博客')
# writer.writerows(ss)
for i in range(100):
    row=('00d91e8e0cca2b76f515926a36db68f5','13641937%03d'%i,'pass%d'%i)
    writer.writerow(row)
csvfile.close()