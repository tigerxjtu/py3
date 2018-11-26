#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: main.py
@time: 2018-10-30
'''

from web.lesson3.gdata import *
from web.lesson3.producer import produce
from web.lesson3.consumer import consume
from web.lesson3.file2queue import readAll
import threading

isStop=stopFlag

def produce_func():
    while produce():
        pass

def consume_func():
    global isStop
    while True:
        flag = consume()
        if (not flag) and isStop:
            break

if __name__=='__main__':
    readAll(r'D:\BaiduNetdiskDownload\Web全栈开发\第三课\sina.txt')
    print(urlQueue.qsize())
    producers=[threading.Thread(target=produce_func) for i in range(3)]
    consumers = [threading.Thread(target=consume_func) for i in range(2)]
    for p in producers:
        p.start()
    for c in consumers:
        c.start()
    for p in producers:
        p.join()
    isStop=True
    for c in consumers:
        c.join()
    print('All done!')