
import csv
# from selenium.webdriver import Remote
from selenium import webdriver
import time


def read_records():
    headers = []
    records = []
    with open('../data.csv', newline='') as csvfile:  # 此方法:当文件不用时会自动关闭文件
        csvReader = csv.reader(csvfile)
        for content in csvReader:
            if not headers:
                headers = content
                continue
            data = {}
            for i, key in enumerate(headers):
                data[key] = content[i]
            records.append(data)
    return records

def login(driver, username, passwd):
    # username_LgA2A
    # password3_LgA2A
    # name = "loginsubmit"
    driver.get('http://www.dataguru.cn/member.php?mod=logging&action=login')
    time.sleep(2)
    driver.find_element_by_css_selector('input[name="username"]').send_keys(username)
    driver.find_element_by_css_selector('input[name="password"]').send_keys(passwd)
    driver.find_element_by_css_selector('button[name="loginsubmit"]').click()
    time.sleep(5)
    account = driver.find_elements_by_partial_link_text(username)
    if account:
        print('login success')
    else:
        print('login failed')


def find_courses(driver, course_name):
    # select_menu('course', '课程')
    driver.get('http://dataguru.cn/search.php?mod=course')
    time.sleep(2)
    # driver.execute_script("select_menu('course', '课程')")
    # scform_srchtxt
    # scform_submit
    driver.find_element_by_id('scform_srchtxt').send_keys(course_name)
    driver.find_element_by_id('scform_submit').click()
    time.sleep(2)
    links=driver.find_elements_by_partial_link_text(course_name)
    if links:
        print('found results')
    else:
        print('not found results')

list_ce = [
    "http://127.0.0.1:4444/wd/hub",
    "http://127.0.0.1:5555/wd/hub",
    "http://127.0.0.1:5556/wd/hub",
    ]
list_dc = [
    {
        "browserName": "chrome",
        "version": "",
        "platform": "ANY",
    },
    {
        "browserName": "internet explorer",
        "version": "",
        "platform": "WINDOWS",
    }
]
'''
    {
        "browserName": "firefox",
        "marionette": True,
        "acceptInsecureCerts": True,
    }
'''


for i in range(len(list_ce)):
    print(list_ce[i])
    print(list_dc[i])
    driver = webdriver.Remote(command_executor=list_ce[i],
    desired_capabilities=list_dc[i])

    # driver = webdriver.Chrome()
    records = read_records()
    for record in records:
        login(driver,**record)

    keywords = ['Python','Pthon']
    for keyword in keywords:
        find_courses(driver,keyword)

    time.sleep(2)
    driver.quit()

