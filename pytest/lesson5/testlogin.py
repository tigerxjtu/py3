#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: testlogin.py
@time: 2018-10-15
'''

from pytest.lesson4.testrequest import *
from pytest.testdata.getpath import GetTestDataPath
import xlrd

_url="http://127.0.0.1:8000"

def test_login():
    url = _url + "/polls/login/"
    testdata = xlrd.open_workbook(GetTestDataPath())
    table = testdata.sheets()[3]
    for i in range(1,3):
        username=table.cell(i,0).value
        pwd = str(int(table.cell(i, 1).value))
        status = str(int(table.cell(i, 2).value))
        hope = table.cell(i, 3).value
        testid="lesson5"+str(i)
        result = TestPostRequest(url, {'username': username, 'password': pwd}, testid, "测试登录", status, hope,
                             None)
    print(result)

test_login()

