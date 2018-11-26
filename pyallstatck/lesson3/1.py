#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: 1.py
@time: 2018-10-06
'''

while True:
    sScore=input('Please input score:')
    score=int(sScore)
    if score>=90:
        print('Excellent')
    elif score>=80:
        print('Good')
    elif score>=60:
        print('Pass')
    else:
        print('Fail')