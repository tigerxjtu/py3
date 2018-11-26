#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: testapi.py
@time: 2018-10-15
'''

from pytest.lesson4.testrequest import *
from pytest.testdata.getpath import GetTestDataPath
import xlrd

def test_novel():
    url = "https://www.apiopen.top/novelSearchApi"
    testdata = xlrd.open_workbook(GetTestDataPath())
    table = testdata.sheets()[4]
    for i in range(1,2):
        name=table.cell(i,0).value
        status = str(int(table.cell(i, 1).value))
        hope = table.cell(i, 2).value
        testid="lesson5"+str(i)
        result = TestGetRequest(url, {'name': name}, testid, "测试小说搜索", status, hope,
                             None,'code')
    print(result)


def test_weather():
    url = "https://www.apiopen.top/weatherApi"
    testdata = xlrd.open_workbook(GetTestDataPath())
    table = testdata.sheets()[4]
    for i in range(4,5):
        city=table.cell(i,0).value
        status = str(int(table.cell(i, 1).value))
        hope = table.cell(i, 2).value
        testid="lesson5"+str(i)
        result = TestGetRequest(url, {'city': city}, testid, "测试天气搜索", status, hope,
                             None,'code')
    print(result)


def test_meitu():
    url = "https://www.apiopen.top/meituApi"
    testdata = xlrd.open_workbook(GetTestDataPath())
    table = testdata.sheets()[4]
    for i in range(7,8):
        page=int(table.cell(i,0).value)
        status = str(int(table.cell(i, 1).value))
        hope = table.cell(i, 2).value
        testid="lesson5"+str(i)
        result = TestGetRequest(url, {'page': page}, testid, "测试美图获取", status, hope,
                             None,'code')
    print(result)

test_novel()
test_weather()
test_meitu()