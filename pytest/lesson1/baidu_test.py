import unittest
from selenium import webdriver
from pytest.lesson1 import baidu


class baidu_test(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()  # 最大化窗口
        self.driver.implicitly_wait(10)  # 隐式等待
        self._baidu = baidu(self.driver)

    def tearDown(self):
        self.driver.quit()

    def test_search(self):
        title = self._baidu.search('python自动化测试')
        print(title)
        self.assertEqual(title, 'python自动化测试_百度搜索')

if __name__ == "__main__":
    unittest.main()