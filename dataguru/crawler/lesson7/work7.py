from selenium.webdriver.common.action_chains import ActionChains
from selenium import webdriver

driver = webdriver.PhantomJS()
url = 'http://book.dangdang.com/'

driver.get(url)

nodes=driver.find_elements_by_xpath('//ul[@id="component_403754__5298_5294__5294"]/li')
for node in nodes:
    title = node.find_element_by_css_selector('p.name>a').text
    price = node.find_element_by_css_selector('p.price>span').text
    print('book title:%s, price:%s'%(title,price))

driver.quit()

