#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: threadtestcase.py
@time: 2018-10-29
'''

from pytest.lesson4.testvote import *
import threading

def threads():
    threads=[]
    threads.append(threading.Thread(target=test_polls))
    threads.append(threading.Thread(target=test_vote))
    threads.append(threading.Thread(target=test_login))
    for th in threads:
        th.start()
    for th in threads:
        th.join()
