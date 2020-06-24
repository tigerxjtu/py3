# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:34:03 2018

@author: Administrator
"""

from selenium import webdriver
import pandas as pd
from urllib.parse import quote
import time
import re




def get_page_comment():
    content=driver.find_elements_by_xpath('//*[@id="comment-0"]//div/div[2]/p') 
    content_list=[c.text for c in content] 
    return content_list

def get_page_all_comment():
    all_content=get_page_comment()
    while True:
        try:
            driver.find_element_by_link_text('下一页').click()
            time.sleep(5)
            all_content=all_content+get_page_comment()
        except:
            break
    return all_content

def get_links(key):
    url='https://search.jd.com/Search?keyword='+quote(key)+'&enc=utf-8'  #构造url
    driver.get(url)  #打开url
    #滚动到页面
    driver.implicitly_wait(3)  #等待

    links=driver.find_elements_by_xpath('//*[@id="J_goodsList"]/ul/li/div/div[1]/a')  #查找当前页面的商品链接
    urls=[l.get_attribute('href') for l in links]  
    return urls

def get_all_comment(urls,outpath='d:/data/jingdong'):
    for url in urls:
        driver.get(url)
        driver.find_element_by_xpath('//*[@id="detail"]/div[1]/ul/li[5]').click() #点击评论
        name=driver.find_element_by_xpath('/html/body/div[8]/div/div[2]/div[1]').text
        comment=get_page_all_comment()
        comment=pd.DataFrame(comment)
        rstr = r"[\/\\\:\*\?\"\<\>\|]"
        name = re.sub(rstr, "_", name)
        comment.to_csv(outpath+name+'.csv')
    return None

def main(key):
    chrome_options = webdriver.ChromeOptions()
    #chrome_options.add_argument('--headless') #设置headless模型
    global driver
    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.maximize_window()
    urls=get_links(key)
    get_all_comment(urls,outpath='d:/data/jingdong')
    driver.close()
    

main('红酒')   
    