#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: test.py
@time: 2018-10-30
'''

# import queue
# q=queue.Queue()
#
# print(q.qsize())
# item=q.get(timeout=1) #queue.Empty error
# if item:
#     print("item:",item)
# else:
#     print('None')

import threading
tl=threading.local()
# tl.name='test'
if tl.__getattribute__('name'):
    print(tl.name)
else:
    tl.name='test'
    print(tl.name)