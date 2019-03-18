#!/usr/bin/python3
# encoding:utf-8

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import time

driver = webdriver.Chrome()
driver.get(r'file:///C:/projects/python/py3/pyallstack2/lesson3/register.html')

ele_mobile=driver.find_element_by_id('mobile')
ele_mobile.send_keys('130000000')
time.sleep(2)

ele_mobile.send_keys(Keys.CONTROL,'a')
ele_mobile.send_keys(Keys.CONTROL,'c')
time.sleep(2)

# ele_mobile.send_keys('123000')
# time.sleep(2)
#
# ele_mobile.clear()
# time.sleep(2)

driver.find_element_by_id('username').send_keys(Keys.CONTROL,'v')
username=driver.find_element_by_id('username').get_attribute('value')
print(username)

driver.find_element_by_id('getcode').click()
code=driver.find_element_by_id('code')
# print(dir(code))
print('code:',code.text)
time.sleep(2)
driver.find_element_by_id('okbtn').click()
print('code:',code.text)
driver.find_element_by_id('verifycode').send_keys(code.text)

checkme=driver.find_element_by_id('checkme')
print('checked:',checkme.is_selected())
# checkme.click()
driver.find_element_by_id('lblcheck').click()
print('checked:',checkme.is_selected())
time.sleep(2)

selected=driver.find_element_by_id('area')
area=Select(selected)
area.select_by_index(2)

time.sleep(2)
area.select_by_visible_text('上海')

time.sleep(2)
area.select_by_value('bj')

time.sleep(2)
ele_cell=driver.find_element_by_xpath("//table[@id='myTable']/tbody/tr[2]/td[2]")
driver.find_element_by_id('username').send_keys(ele_cell.text)
print(ele_cell.text)
time.sleep(2)

def getCell(driver,tableId,row,col):
    xpath="//table[@id='{}']/tbody/tr[{}]/td[{}]".format(tableId,row,col)
    ele_cell = driver.find_element_by_xpath(xpath)
    return ele_cell.text

driver.find_element_by_id('username').clear()
driver.find_element_by_id('username').send_keys(getCell(driver,'myTable',3,2))

time.sleep(5)
driver.quit()