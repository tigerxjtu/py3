# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:42:12 2019

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

df = pd.read_excel('朝阳医院2018年销售数据.xlsx')
df['购药时间'].dropna(inplace=True)
#pd.to_datetime(df['购药时间'],format='%Y-%M-%d')
from datetime import datetime
def to_month(s):
    day=s.split()[0]
    date=datetime.strptime(day,'%Y-%M-%d')
    return date.strftime('%Y-%M')

df['购药时间']=df['购药时间'].apply(to_month)
grouped = df[['应收金额','销售数量']].groupby(df['购药时间'])
data=grouped.sum()
rows=len(data)


fig, ax1 = plt.subplots() #创建画图对象
ax1.plot(range(rows),data['应收金额'],c='b',ls='--',label='销售额')
ax1.legend(loc='upper right')
ax1.set_xticks(range(rows))  # 位置
ax1.set_xticklabels(data.index,rotation=45)
ax1.xaxis.set_label_coords(1.05,-0.02)
ax1.set_xlabel('日期',fontsize=10)
ax1.set_ylabel('销售额',color='b')
ax1.tick_params('y',colors='b') #使坐标轴的和线条相匹配
ax2 = ax1.twinx() #使用子坐标
ax2.plot(range(rows),data['销售数量'],c='r',ls=':',label='销售量')
ax2.set_xticks(range(rows))  # 位置
#ax2.set_xticklabels(jd_stock.iloc[range(0,70,4),1],rotation=45)
ax2.legend(loc='upper left')
ax2.set_xlabel('日期',fontsize=10)
ax2.set_ylabel('销售量',color='r')
ax2.tick_params('y',colors='r') #使坐标轴的和线条相匹配
plt.show()

#将朝阳医院不同商品的销售数量进行汇总
df = pd.read_excel('朝阳医院2018年销售数据.xlsx')
grouped = df['销售数量'].groupby(df['商品名称'])
data=grouped.sum()
#data=pd.DataFrame({'商品名称':data.index,'销售数量':data})
plt.figure(figsize=(32,16))
plt.bar(range(len(data.index)),data,width=0.6,align='center')
plt.xlabel("药品名称",fontsize=18)
plt.ylabel('销售数量',fontsize=18)
plt.xticks(range(len(data.index.values)),data.index.values,rotation=90)  # 位置
plt.title('药品销量')
plt.show()

#绘制不同edu_class下的 avg_exp和Income的关系图
df=pd.read_csv('creditcard_exp.csv')
df['edu_class'].value_counts()

p=ggplot(aes(x='avg_exp',y='Income',group='factor(edu_class)'),data=df) + geom_point(size=12) \
  + facet_wrap('factor(edu_class)') + labs(x='平均支出', y='收入', title='不同edu_class下的 avg_exp和Income的关系图')
print (p)

