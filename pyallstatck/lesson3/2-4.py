#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: 2-4.py
@time: 2018-10-06
'''

print('2、写一个简单的python小程序，用字典保存你的计算机的型号，操作系统，内存大小，CPU类型，并使用不同的方法遍历输出；')
computer={'model':'dell e7410','os':'Win10','Memory':'16G','cpu':'intel i7'}
for k in computer:
    print(k,computer[k])
for k,v in computer.items():
    print(k,v)

print('3、对上面的字典添加你的显卡类型，修改内存为32G，删除显卡类型；')
computer['Graphics card']='Nivida 1080ti'
computer['Memory']='32G'
for k,v in computer.items():
    print(k,v)
del(computer['Graphics card'])
print('after remove Graphics card:')
for k,v in computer.items():
    print(k,v)

print('4、新建一个空的集合，并将上面的字典里的key添加进来。')
keys=set()
keys.update(computer.keys())
print(keys)