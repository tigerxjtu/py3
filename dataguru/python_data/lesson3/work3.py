# Q1 请读取meal_order_detail.xlsx文件，将其所有工作表进行合并，然后保存在csv里面。
import json
import os
import pandas as pd
import xlrd

path = r"C:\dataguru_new\Python数据处理实战：基于真实场景的数据\第三课\数据和代码"
file = os.path.join(path, 'meal_order_detail.xlsx')
all_df = pd.DataFrame()
workbook = xlrd.open_workbook(file)
sheet_names = workbook.sheet_names()
for sheet_name in sheet_names:
    df = pd.read_excel(file, sheet_name=sheet_name)
    all_df = pd.concat([all_df, df], axis=0)
all_df.info()
all_df.reset_index(drop=True, inplace=True)
all_df.to_csv(os.path.join(path, 'result.csv'), index=False)
new_df = pd.read_csv(os.path.join(path, 'result.csv'), engine='python')
new_df.info()

# Q2 请读取equipment_1.txt文件,  请使用课上的两种方法。
file = os.path.join(path, 'equipment_1.txt')
# 方法1
df = pd.read_csv(file, sep='\t', engine='python', encoding='utf-8')
df.head(5)
# 方法2
with open(file, encoding='utf-8') as f:
    f_read = f.readlines()
data = []
for line in f_read[1:]:
    line = line.split('\t')
    line = [i.strip('\n') for i in line]
    data.append(line)
header = f_read[0].split('\t')  # 字段名称
header[-1] = header[-1].strip('\n')
df = pd.DataFrame(data, columns=header)
df.head(5)

# Q3 (选做)请自行安装mysql和客户端, 将meal_order_detail.xlsx合并后的数据保存进去。(自己电脑上密码,用户名自行设置就好)
import pymysql
from sqlalchemy import create_engine
# conn = create_engine('mysql+pymysql://root:pass@localhost:3306/test', encoding='utf-8')
conn = create_engine('mysql+pymysql://root:pass@192.168.20.83:3306/test', encoding='utf-8')
new_df.to_sql('testdf',con = conn, index= False,if_exists= 'replace')

df1=pd.read_sql("select * from testdf",conn)
df1.head(10)
df1[['detail_id','dishes_name']].head(10)