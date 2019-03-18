#!/usr/bin/python3
# encoding:utf-8

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time

driver = webdriver.Chrome()
driver.get(r'http://www.runoob.com/try/try.php?filename=tryjsref_ondrag_all')

driver.switch_to_frame('iframeResult')
ele_src=driver.find_element_by_id('dragtarget')
elements=driver.find_elements_by_class_name('droptarget')
print(len(elements))
# ele_src=elements[0]
ele_dst=elements[1]
# ActionChains(driver).move_to_element(ele_src)
# ele_src.click()
ActionChains(driver).drag_and_drop(ele_src,ele_dst).perform()

time.sleep(5)
driver.quit()