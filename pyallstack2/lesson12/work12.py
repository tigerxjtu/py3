import unittest
import ddt
import os
from pyallstack2.lesson10.test_login import read_records , login
# import pytest.lesson2.HTMLTestRunner as HTMLTestRunner
import pyallstack2.lesson12.HTMLTestRunner as HTMLTestRunner
import pyallstack2.lesson11.test_myadd as test_myadd
records=read_records()
print(records)

@ddt.ddt
class Work12TestCase(unittest.TestCase):
    @ddt.data(*records)
    @ddt.unpack
    def test_login(self, username,password,result):
        print('test')
        payload = dict(username=username, password=password)
        ret=login(payload)
        self.assertEqual(ret,int(result))

if __name__ == '__main__':
    # testSuite = unittest.TestSuite()
    # testSuite.addTest(Work12TestCase())
    testSuite = unittest.defaultTestLoader.discover(os.getcwd(),pattern='work12.py')
    # testSuite.addTest(test_myadd.TestMyAdd())
    # runner = unittest.TextTestRunner()
    # runner.run(testSuite)
    fp=open('result.html','wb')
    runner = HTMLTestRunner.HTMLTestRunner(stream=fp, title='work12', description='api login test')
    runner.run(testSuite)
    fp.close()
