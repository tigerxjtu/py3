#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: test_lesson2.py
@time: 2018-09-18
'''

from pytest.lesson2.HTMLTestRunner import HTMLTestRunner
import unittest
from pytest.lesson2.lesson2 import Lesson2Test

suit = unittest.TestSuite()
suit.addTest(Lesson2Test("test_url1"))
suit.addTest(Lesson2Test("test_url2"))
suit.addTest(Lesson2Test("test_url3"))
suit.addTest(Lesson2Test("test_url4"))
suit.addTest(Lesson2Test("test_url5"))

fp = open('d:/result.html', 'wb')
runner = HTMLTestRunner(stream=fp,
                        title='api test report',
                        description='api test report for lesson2 ')

runner.run(suit)
fp.close()
