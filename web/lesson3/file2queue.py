#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: file2queue.py
@time: 2018-10-30
'''

from web.lesson3.gdata import urlQueue

q=urlQueue

def readAll(file):
    with open(file,'r') as f:
        for line in f:
            q.put(line)