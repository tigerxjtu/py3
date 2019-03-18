from pyallstack2.lesson7.BasePage import BasePage
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

class RegisterPage(BasePage):
    base_url='http://dataguru.cn/member.php?mod=register'
    submit_loc=(By.ID,'registerformsubmit')

    def __init__(self):
        self.driver=webdriver.Chrome()
        super().__init__(self.driver,self.base_url)
        self.open(self.base_url)


    def fill_form(self,username,pwd,email):
        xpath="//div[@id='reginfo_a']//td/input"
        elements=self.driver.find_elements_by_xpath(xpath)
        if (not elements) and len(elements)<5:
            print('定位表单元素失败')
            return False
        # print(type(elements))
        elements[0].send_keys(username)
        time.sleep(1)
        elements[1].send_keys(pwd)
        time.sleep(1)
        elements[2].send_keys(pwd)
        time.sleep(1)
        elements[3].send_keys(email)
        time.sleep(1)
        return True

    def submit_form(self):
        ele_submit=self.base_find_element(*self.submit_loc)
        if not ele_submit:
            print('定位提交按钮失败')
            return False
        ele_submit.click()
        return True


page=RegisterPage()
result=page.fill_form('tigerxjtu111','111@tigerxjtu','smple@qq.com')
if result:
    time.sleep(20)
    page.submit_form()
time.sleep(2)
page.quit()
