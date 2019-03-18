#!/usr/bin/python3
# encoding:utf-8

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time

driver = webdriver.Chrome()
driver.get(r'file:///C:/projects/python/py3/pyallstack2/lesson3/register.html')

ele_getcode=driver.find_element_by_id('getcode')
ActionChains(driver).context_click(ele_getcode).perform()
time.sleep(2)
ele_getcode.click()



time.sleep(5)
driver.quit()