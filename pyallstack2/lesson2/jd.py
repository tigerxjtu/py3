#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: lesson2.py
@time: 2019-01-29
'''

from selenium import webdriver
import time

driver = webdriver.Chrome()
driver.maximize_window()  # 最大化窗口
driver.get('https://list.jd.com/list.html?cat=652,654,831')

eles=driver.find_elements_by_xpath('//*[@id="plist"]/ul/li')
for sel in eles:
    ele = sel.find_element_by_xpath('.//div[@class="p-price"]//i')
    print(ele.text)
    ele = sel.find_element_by_xpath('.//div[@class="p-name"]//em')
    print(ele.text)


time.sleep(5)
driver.quit()