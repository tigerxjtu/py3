from selenium import webdriver
import pandas as pd
from urllib.parse import quote
import json
import time

chrome_options = webdriver.ChromeOptions()
driver = webdriver.Chrome(chrome_options=chrome_options)
driver.maximize_window()

key='iphone'
url='https://search.jd.com/Search?keyword='+quote(key)+'&enc=utf-8'  #构造url
driver.get(url)  #打开url
#滚动到页面
driver.implicitly_wait(3)  #等待

def get_products(driver):
    # // *[ @ id = "J_goodsList"] / ul / li[1]
    results=[]
    elements=driver.find_elements_by_xpath('//div[@id="J_goodsList"]/ul/li')
    for e in elements[:10]:
        product={}
        product['name']=e.find_element_by_xpath('.//div[contains(@class,"p-name")]/a/em').text
        product['url'] = e.find_element_by_xpath('.//div[contains(@class,"p-name")]/a').get_attribute('href')
        # product['p-shop'] = e.find_element_by_xpath('.//div[@class="p-shop"]/a').text
        product['price'] = e.find_element_by_xpath('.//div[@class="p-price"]/strong/i').text
        results.append(product)
        print(product)
    return results

def get_page_comment(driver):
    content=driver.find_elements_by_xpath('//*[@id="comment-0"]//div/div[2]/p')
    content_list=[c.text for c in content]
    return content_list

def get_page_all_comment(driver):
    all_content=get_page_comment()
    while True:
        try:
            driver.find_element_by_link_text('下一页').click()
            time.sleep(5)
            all_content=all_content+get_page_comment()
        except:
            break
    return all_content

def get_all_comment(url,driver):
    driver.get(url)
    driver.find_element_by_xpath('//*[@id="detail"]/div[1]/ul/li[5]').click() #点击评论
    name=driver.find_element_by_xpath('/html/body/div[8]/div/div[2]/div[1]').text
    comment=get_page_all_comment(driver)
    return comment

# //*[@id="J_filter"]/div[1]/div[1]/a[2]
driver.find_element_by_xpath('//*[@id="J_filter"]/div[1]/div[1]/a[2]').click() #点击销量
driver.implicitly_wait(8)  #等待
products=get_products(driver)
try:
    for product in products:
        product['comment']=get_all_comment(product['url'])
finally:
    with open('comments.json','w') as f:
        json.dump(products,f)
driver.quit()