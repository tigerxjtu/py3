from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException,TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import time

driver = webdriver.Chrome()
# driver.implicitly_wait(10)

driver.get(r'file:///C:/projects/python/py3/pyallstack2/lesson4/register.html')

# driver.find_element_by_id('btnAlert').click()
# time.sleep(2)
# driver.switch_to.alert.accept()

# driver.find_element_by_id('btnConfirm').click()
# time.sleep(2)
# driver.switch_to.alert.dismiss()

# driver.find_element_by_id('btnPrompt').click()
# time.sleep(2)
# alert_prompt=driver.switch_to.alert
# alert_prompt.send_keys('hello')
# print(alert_prompt.text)
# alert_prompt.accept()
driver.execute_script('show()')
time.sleep(2)

driver.switch_to.frame('iframe')
ele_mobile=driver.find_element_by_id('mobile')
ele_mobile.send_keys('130000000')

value=driver.execute_script('return document.getElementById("mobile").value')
print(value)

# driver.switch_to_default_content()
driver.switch_to.parent_frame()

current_win=driver.current_window_handle
driver.find_element_by_link_text('Baidu').click()

windows=driver.window_handles
for w in windows:
    # print(w.title(),type(w))
    # print(dir(w))
    if w!=current_win:
        print('in baidu window')
        driver.switch_to.window(w)
        driver.find_element_by_id('kw').send_keys('python')

driver.switch_to.window(current_win)
driver.execute_script('alert("'+value+'")')
time.sleep(2)
driver.switch_to.alert.accept()

driver.save_screenshot('lesson4.png')
time.sleep(5)
driver.quit()
