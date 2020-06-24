from selenium import webdriver
import pandas as pd
from urllib.parse import quote
import json

chrome_options = webdriver.ChromeOptions()
driver = webdriver.Chrome(chrome_options=chrome_options)
driver.maximize_window()

key='python'
url='https://search.jd.com/Search?keyword='+quote(key)+'&enc=utf-8'  #构造url
driver.get(url)  #打开url
#滚动到页面
driver.implicitly_wait(3)  #等待

# 商品名称、链接、店铺等
def get_products(driver):
    # // *[ @ id = "J_goodsList"] / ul / li[1]
    results=[]
    elements=driver.find_elements_by_xpath('//div[@id="J_goodsList"]/ul/li')
    for e in elements:
        product={}
        product['name']=e.find_element_by_xpath('.//div[@class="p-name"]/a/em').text
        product['url'] = e.find_element_by_xpath('.//div[@class="p-name"]/a').get_attribute('href')
        # product['shop'] = e.find_element_by_xpath('.//div[@class="p-shopnum"]/a').text
        product['price'] = e.find_element_by_xpath('.//div[@class="p-price"]/strong/i').text
        results.append(product)
        print(product)
    return results

# //*[@id="J_filter"]/div[1]/div[1]/a[2]
# driver.find_element_by_xpath('//*[@id="J_filter"]/div[1]/div[1]/a[2]').click() #点击销量
products=get_products(driver)
pages=driver.find_element_by_xpath('//*[@id="J_bottomPage"]/span[2]/em[1]/b').text
pages=int(pages)
try:
    for i in range(2,pages+1):
        url='https://search.jd.com/Search?keyword=python&enc=utf-8&page='+str(i)
        driver.get(url)  # 打开url
        # 滚动到页面
        driver.implicitly_wait(3)  # 等待
        products=products+get_products(driver)
finally:
    with open('products.json','w') as f:
        json.dump(products,f)

driver.quit()