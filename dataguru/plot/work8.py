# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:42:37 2019

@author: liyin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.chdir(r'C:\data')
pd.set_option('display.max_columns',8)

import seaborn as sns
sns.set(style='darkgrid',context='notebook',font_scale=1.5) # 设置背景

taobao = pd.read_csv('taobao_data.csv')
#taobao['total']=taobao['价格']*taobao['成交量']

taobao_grouped=taobao[['价格','成交量']].groupby(taobao['位置'])
taobao_sum=taobao_grouped.sum()
taobao_mean=taobao_grouped.mean()

taobao_sum['type']='总数'
taobao_mean['type']='均值'

data=pd.concat([taobao_sum,taobao_mean])
data=data.reset_index()


fig = plt.figure(figsize=(16,9))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.subplot2grid((2,1),(0,0))
sns.barplot(x = '位置',y = '价格',hue='type',data=data, palette="husl",
            orient ='vertical')
plt.ylabel('价格',fontsize=20)
plt.xlabel('')
plt.title('不同省份价格的总数和均值情况',fontsize = 25)

plt.subplot2grid((2,1),(1,0))
sns.barplot(x = '位置',y = '成交量',hue='type',data=data, palette="husl",
            orient ='vertical')
plt.ylabel('价格',fontsize=20)
plt.xlabel('')
plt.title('不同省份销量的总数和均值情况',fontsize = 25)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()


df = pd.read_excel('朝阳医院2018年销售数据.xlsx')
#total=df['应收金额'].sum()
#grouped = df['应收金额'].groupby(df['商品名称'])
#data=grouped.sum()

from datetime import datetime
df['购药时间'].dropna(inplace=True)
def to_month(s):
    day=s.split()[0]
    date=datetime.strptime(day,'%Y-%M-%d')
    return date.strftime('%Y-%M')
df['购药时间']=df['购药时间'].apply(to_month)
grouped = df[['应收金额','销售数量']].groupby(df['购药时间'])
data = grouped.sum()

sns.scatterplot(x = '销售数量',y = '应收金额',data = data,color="red", marker='o', s=40)
plt.xlabel('销售数量')
plt.ylabel('应收金额')
plt.title('销售数量和应收金额关系图')
plt.show()

