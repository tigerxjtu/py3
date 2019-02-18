#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: lesson2.py
@time: 2019-01-29
'''

from selenium import webdriver
import time

driver = webdriver.Chrome()
driver.maximize_window()  # 最大化窗口
driver.get('https://www.jd.com/')

#s首页登录链接
login_href=driver.find_element_by_class_name('link-login')
print(login_href.text)
login_href.click()

account_login=driver.find_element_by_partial_link_text("账户登录")
account_login.click()

username_input=driver.find_element_by_id("loginname")
username_input.send_keys('tigerxjtu')

passwd_input=driver.find_element_by_css_selector('input[type="password"]')
passwd_input.send_keys('********')

submit_btn=driver.find_element_by_css_selector('div.login-btn>a')
submit_btn.click()


time.sleep(5)
driver.quit()