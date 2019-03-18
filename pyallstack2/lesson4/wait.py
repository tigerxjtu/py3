from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException,TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import time

def getCell(driver,tableId,row,col):
    xpath="//table[@id='{}']/tbody/tr[{}]/td[{}]".format(tableId,row,col)
    ele_cell = driver.find_element_by_xpath(xpath)
    return ele_cell

driver = webdriver.Chrome()
# driver.implicitly_wait(10)

driver.get(r'file:///C:/projects/python/py3/pyallstack2/lesson4/register.html')
try:
    print(time.ctime())
    # para=getCell(driver,'myTable',3,2)
    # para=driver.find_element_by_id('hello1')
    # para = driver.find_element_by_id('hello')
    # para = WebDriverWait(driver,5,0.5).until(EC.presence_of_element_located((By.ID,'hello1')))
    para = WebDriverWait(driver, 5, 0.5).until(EC.presence_of_element_located((By.ID, 'hello')))
    txt=para.text
    print(time.ctime())
    print('para:',txt)
except NoSuchElementException as e:
    print(e)
    print(time.ctime())
except TimeoutException as e:
    print(e)
    print(time.ctime())

time.sleep(5)
driver.quit()
