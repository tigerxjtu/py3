# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 23:13:15 2018

@author: lenovo-pc
"""

from selenium import webdriver
import pandas as pd
from urllib.parse import quote

driver = webdriver.Chrome() #打开浏览器
key='红酒' #设置搜索商品关键词

url='https://search.jd.com/Search?keyword='+quote(key)+'&enc=utf-8'  #构造url

driver.get(url)  #打开url
driver.implicitly_wait(3)  #等待

links=driver.find_elements_by_xpath('//*[@id="J_goodsList"]/ul/li/div/div[3]/a')  #查找当前页面的商品链接
urls=[l.get_attribute('href') for l in links]  

url=urls[1] #获取第一个商品链接
#//*[@id="comment-0"]/div[1]/div[2]/p
#//*[@id="comment-0"]/div[3]/div[2]/p
driver.get(url) #打开页面
driver.find_element_by_xpath('//*[@id="detail"]/div[1]/ul/li[5]').click() #点解商品评论
#driver.find_element_by_xpath('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[1]/a').click()  #点解评论

#获取评论数据


content=driver.find_elements_by_xpath('//*[@id="comment-0"]//div/div[2]/p') 
content_list2=[c.text for c in content] 

driver.find_element_by_link_text('下一页').click() #点击下一页评论









