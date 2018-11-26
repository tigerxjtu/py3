# -*-coding:utf-8-*-
from threading import Thread, Lock
import os
import time

lock = Lock()

f = open('d:/data/test.txt','a+')

fr = open('d:/data/test.txt','r')

def  writeFile(n):
    global  f

    for i in range(10):
        lock.acquire()
        line='line{} in write thread{}\n'.format(i+1, n)
        print('write line:{}'.format(line))
        f.write(line)
        f.flush()
        lock.release()
        time.sleep(0.1)

def readFile(n):
    global f

    for i in range(15):
        lock.acquire()
        try:
            line = fr.readline()
            if line=='':
                continue
            print('read in thread{}:{}'.format(n,line))
        except:
            print('error')
        finally:
            lock.release()
        time.sleep(0.1)

def start():
    wts=[Thread(target=writeFile, args=(i,)) for i in range(5)]
    rts=[Thread(target=readFile, args=(i,)) for i in range(5)]
    [t.start() for t in wts]
    [t.start() for t in rts]
    [t.join() for t in wts]
    [t.join() for t in rts]
    f.close()
    fr.close()

start()
# for i in range(10):
#     lock.acquire()
#     try:
#         line = fr.readline()
#         print('read:{}'.format(line))
#     except:
#         print('error')
#     time.sleep(0.1)
#     lock.release()