#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: exam.py
@time: 2018-12-17
'''
def swap(list):
    temp=list[0]
    list[0]=list[1]
    list[1]=temp
list=[1,2]
swap(list)
print(list)
