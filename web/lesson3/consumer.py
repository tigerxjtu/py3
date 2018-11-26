#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: consumer.py
@time: 2018-10-30
'''

from web.lesson3.gdata import *
import pymysql
import threading

qData=dataQueue

threaddb=threading.local()


def get_connection():
    try:
        return threaddb.db
    except AttributeError:
        db = pymysql.connect(user='root', db='test', passwd='pass', host='127.0.0.1', use_unicode=True, charset='utf8')
        threaddb.db=db
        return threaddb.db

def save(item):
    sql='insert into sina (url,title,content) values (%s,%s,%s)'
    db=get_connection()
    with db.cursor() as cur:
        cur.execute(sql,item)
    db.commit()
    # db.close()

def consume():
    global qData
    try:
        item=qData.get(timeout=10)
        print("consume:",item[0])
        print('saving...')
        save(item)
        print('saving done')
        return True
    except pymysql.err.InternalError as e:
        print('数据库访问错误：',e.args)
        return False
    except queue.Empty as e:
        return False