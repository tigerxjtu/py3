#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: test11.py
@time: 2019-01-21
'''
import re

x=43
ch='A'
y = 1
# print(x>=y and ch <'b' and y)

s=''' {"code":200,"msg":"成功!","data":{"key":"00d91e8e0cca2b76f515926a36db68f5","phone":"13594347817","name":"peakchao","passwd":"123456","text":"这是我的签名。","img":"https://res.apiopen.top/2018031820405521.key.png","other":"这是我的备注信息1","other2":"这是我的备注信息2","createTime":"2018-03-18 20:40:55"}}'''
regexp = r'.*{"key":"(?P<key>[a-z0-9]+)".*'
reg = re.compile(regexp)
m=reg.match(s)
key=m.group('key')
print(key)