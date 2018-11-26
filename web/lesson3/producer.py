#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: producer.py
@time: 2018-10-30
'''

from web.lesson3.gdata import *
from web.lesson3.file2queue import readAll
import requests
import bs4

qUrl=urlQueue
qData=dataQueue

def produce():
    global qUrl
    global qData
    try:
        url=qUrl.get(timeout=2)
        print(url)
        item=crawl(url)
        qData.put(item)
        return True
    except:
        return False

def crawl(url):
        result=requests.get(url)
        content=result.content.decode('gbk')
        return url,get_title(content),content

def get_title(text):
    soup = bs4.BeautifulSoup(text, "lxml")
    try:
        return soup.find_all('title')[0].get_text()
    except:
        return ''
