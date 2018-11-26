#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: mysum.py
@time: 2018-10-18
'''

from functools import reduce

def mysum(n):
    return reduce(lambda x,y:x+y,(i**2 for i in range(1,n+1)))

if __name__ == '__main__':
    for i in range(2,10):
        print(mysum(i))