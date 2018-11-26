#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: mycall.py
@time: 2018-10-18
'''

import time
from pyallstatck.lesson4.mylib.mysum import mysum


def mycall(n):
    print("当前时间："+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print("现在正执行mycall函数")

    return mysum(n)

if __name__ == '__main__':
    print(mycall(10))