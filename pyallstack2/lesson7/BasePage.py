from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import os
import time


class BasePage():
    #页面信息初始化
    def __init__(self,driver,url):
        self.driver = driver
        self.url = url

    # 定义url获取，如果传入url就使用传入的url，未传入使用上面的默认url
    def open(self,url=None):
        self.driver.maximize_window() # 最大化窗口
        self.driver.implicitly_wait(10)# 隐式等待
        if url is None:
            self.driver.get(self.url)
        else:
            self.driver.get(url)

    # 判断页面是否打开成功
    def assert_page_by_title(self, title):
        cur_title = self.driver.title
        if title.lower() == cur_title.lower():
            return True
        else:
            return False

    # 定位元素
    def base_find_element(self, *loc, timeout=5, frequency=0.5):
        try:
            element = WebDriverWait(self.driver, timeout, frequency).until(
                EC.presence_of_element_located(loc)
            )
        except TimeoutException as e:
            print(e)
            return None
        else:
            return element

    # 执行js脚本
    def base_execute_script(self, str_script):
        return self.driver.execute_script(str_script)

    # 保存截图
    def save_picture(self, pic_name):
        time.sleep(3)
        # 文件名保存不应该含有”:”等特殊符号否则文件无法保存成功
        name = pic_name + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + ".png"
        str_path = os.path.join(os.getcwd(), "testPicture", name)
        print(str_path)
        self.driver.get_screenshot_as_file(str_path)

    def quit(self):
        self.driver.quit()

