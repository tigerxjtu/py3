#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: testdatademo.py
@time: 2018-10-15
'''

import os
import xlrd

def GetTestDataPath():
    ospath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(ospath,"testdata","TestData.xls")

testdata=xlrd.open_workbook(GetTestDataPath())
table=testdata.sheets()[1]

choice=table.cell(3,0).value
status=table.cell(3,1).value
print(choice)
print(status)