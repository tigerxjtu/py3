import os
import time
import xlrd

def GetTestDataPath():
    ospath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(ospath,"testdata","TestData.xls")

# testdata=xlrd.open_workbook(GetTestDataPath())
# table=testdata.sheet()[1]
#
# choice=table.cell(3,0).value
# print(choice)

def GetTestReport():
    ospath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    now=time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
    return os.path.join(ospath, "testreport", now+"TestReport.xls")


def GetTestLogPath():
    ospath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(ospath, "logs","logs.txt")

def GetConfigPath(config_file):
    ospath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(ospath, "conf",config_file)


