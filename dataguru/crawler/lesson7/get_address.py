# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:53:44 2018

@author: Administrator
"""

from selenium.webdriver.common.action_chains import ActionChains
from selenium import webdriver
import pandas as pd
import re

driver = webdriver.PhantomJS(executable_path=r'D:/Program Files/phantomjs-2.1.1-windows/bin/phantomjs')

url='https://map.baidu.com/'

driver.get(url)  

company='广东欧珀移动通信有限公司'
input_node=driver.find_element_by_xpath('//*[@id="sole-input"]')

input_node.send_keys(company)
#input_node.submit()

#from selenium.webdriver.common.keys import Keys
#input_node.send_keys(Keys.ENTER)
driver.find_element_by_xpath('//*[@id="search-button"]').click()
driver.implicitly_wait(3)
#driver.refresh() 
driver.find_element_by_xpath('//*[@id="card-1"]/div/ul/li[1]/div[1]/div[3]/div[2]').text

driver.refresh()
company='广州酒家'
input_node=driver.find_element_by_xpath('//*[@id="sole-input"]')

input_node.send_keys(company)
#input_node.submit()

#from selenium.webdriver.common.keys import Keys
#input_node.send_keys(Keys.ENTER)
driver.find_element_by_xpath('//*[@id="search-button"]').click()
driver.implicitly_wait(3)
#driver.refresh() 
driver.find_element_by_xpath('//*[@id="card-1"]/div/ul/li[1]/div[1]/div[3]/div[3]').text


driver.close()


driver = webdriver.PhantomJS(executable_path=r'D:/Program Files/phantomjs-2.1.1-windows/bin/phantomjs')

url='http://api.map.baidu.com/lbsapi/getpoint/'

driver.get(url)  

company='广东欧珀移动通信有限公司'

input_node=driver.find_element_by_xpath('//*[@id="localvalue"]')
input_node.send_keys(company)

driver.find_element_by_xpath('//*[@id="localsearch"]').click()
driver.implicitly_wait(3)

driver.find_element_by_xpath('//*[@id="no_0"]/p').text.split('\n')



def get_address(company):
    #driver.refresh()
    input_node=driver.find_element_by_xpath('//*[@id="localvalue"]')
    input_node.send_keys(company)
    
    driver.find_element_by_xpath('//*[@id="localsearch"]').click()
    driver.implicitly_wait(3)
    
    result=driver.find_element_by_xpath('//*[@id="no_0"]/p').text.split('\n')
    
    return result

driver = webdriver.PhantomJS(executable_path=r'D:/Program Files/phantomjs-2.1.1-windows/bin/phantomjs')

url='http://api.map.baidu.com/lbsapi/getpoint/'

driver.get(url)



