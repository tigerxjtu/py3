#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: lesson2.py
@time: 2019-01-29
'''

from lxml import etree
import requests

furl=requests.get('https://list.jd.com/list.html?cat=652,654,831')
tree = etree.HTML(furl.content)

eles=tree.xpath('//*[@id="plist"]/ul/li')
for sel in eles:
    ele = sel.xpath('.//div[@class="p-name"]//em/text()')
    ele = sel.xpath('.//div[@class="p-price"]/text()')
    print(ele[0])
    ele = sel.xpath('.//div[@class="p-price"]//i/text()')
    print(ele[0])


