#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: exceptiontest.py
@time: 2018-10-30
'''

def file_except():
    path='test/notexist.txt'
    try:
        fp=open(path,'r')
    except IOError as e:
        print('文件不存在！',e.args)
    else:
        fp.close()

def arithmetic_except():
    a=1
    b=0
    try:
        return a/b
    except ArithmeticError as e:
        print("运算错误：",e.args)

def input_except():
    num = input(">>: ")
    try:
        return int(num)
    except ValueError as e:
        print('请输入整数：',e.args)
        return input_except()

file_except()
arithmetic_except()
print(input_except())


