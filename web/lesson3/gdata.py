#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: gdata.py
@time: 2018-10-30
'''

import queue

urlQueue=queue.Queue()
dataQueue=queue.Queue(5)

stopFlag=False
