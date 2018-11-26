#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: lesson2.py
@time: 2018-09-18
'''

from pytest.lesson2.HTMLTestRunner import HTMLTestRunner
import unittest
import json
import requests


class Lesson2Test(unittest.TestCase):

    def get_url(self, url):
        r=requests.get(url)
        return (r.status_code, json.loads(r.text))

    def test_url1(self):
        url="https://www.apiopen.top/journalismApi"
        code,obj=self.get_url(url)
        self.assertEqual(code,200)
        self.assertEqual(obj['code'],200)

    def test_url2(self):
        url="https://www.apiopen.top/satinGodApi?type=1&page=1"
        code,obj=self.get_url(url)
        self.assertEqual(code,200)
        self.assertEqual(obj['code'],200)

    def test_url3(self):
        url="https://www.apiopen.top/novelApi"
        code,obj=self.get_url(url)
        self.assertEqual(code,200)
        self.assertEqual(obj['code'],200)

    def test_url4(self):
        url="https://www.apiopen.top/novelSearchApi?name=%E7%9B%97%E5%A2%93%E7%AC%94%E8%AE%B0"
        code,obj=self.get_url(url)
        self.assertEqual(code,200)
        self.assertEqual(obj['code'],200)

    def test_url5(self):
        url="https://www.apiopen.top/meituApi?page=1"
        code,obj=self.get_url(url)
        self.assertEqual(code,200)
        self.assertEqual(obj['code'],200)


if __name__ == '__main__':
    suit=unittest.TestSuite()
    suit.addTest(Lesson2Test("test_url1"))
    suit.addTest(Lesson2Test("test_url2"))
    suit.addTest(Lesson2Test("test_url3"))
    suit.addTest(Lesson2Test("test_url4"))
    suit.addTest(Lesson2Test("test_url5"))

    fp = open('d:/result.html', 'wb')
    runner =HTMLTestRunner(stream=fp,
                                          title='api test report',
                                          description='api test report for lesson2 ')

    runner.run(suit)
    fp.close()