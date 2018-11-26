from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

class baidu():
    def __init__(self, driver):
        self._driver=driver #不能在类中再次导入webdriver 两边的driver等于两个窗口，直接让调用方传入driver即可

    def search(self,keyword):
        dr=self._driver
        dr.get('https://www.baidu.com/')
        dr.find_element_by_xpath("//*[@id='kw']").send_keys(keyword)
        dr.find_element_by_xpath("//*[@id='su']").click()
        wait = WebDriverWait(dr, 10)  # 等待元素加载出来
        wait.until(EC.presence_of_element_located((By.ID,'content_left')))  # 加载
        return dr.title
