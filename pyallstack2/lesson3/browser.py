#!/usr/bin/python3
# encoding:utf-8

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time

driver = webdriver.Chrome()
driver.get('https://www.baidu.com')
driver.maximize_window()
time.sleep(2)

driver.find_element_by_id('kw').send_keys('python')
driver.find_element_by_id('su').click()
time.sleep(2)

driver.back()
time.sleep(2)

driver.forward()
time.sleep(2)

driver.refresh()

time.sleep(5)
driver.quit()
