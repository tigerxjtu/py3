# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:55:33 2019

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
total=df['应收金额'].sum()
grouped = df['应收金额'].groupby(df['商品名称'])
data=grouped.sum()

fig, ax = plt.subplots(2,1,sharex=True,figsize=(16,9))

plot = ax[0]

plot.bar(range(len(data.index.values)),data,width=0.6,align='center')
plot.set_ylabel('销售额',fontsize=18)

plot = ax[1]
grouped = df['销售数量'].groupby(df['商品名称'])
data=grouped.sum()
plot.bar(range(len(data.index.values)),data,width=0.6,align='center')
plot.set_xlabel("药品名称",fontsize=18)
plot.set_ylabel('销售数量',fontsize=18)
plot.set_xticks(range(len(data.index.values)))  # 位置
plot.set_xticklabels(data.index.values,rotation=45)
plt.suptitle('药品销量')
plt.show()

cars = pd.read_csv('sec_cars.csv')
taobao = pd.read_csv('taobao_data.csv')
taobao['total']=taobao['价格']*taobao['成交量']

cars_grouped=cars['Sec_price'].groupby(cars['Brand'])
cars_price=cars_grouped.mean()

taobao_total=taobao['total'].groupby(taobao['位置']).sum()
taobao_qty=taobao['成交量'].groupby(taobao['位置']).sum()

#非等分画布
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(121)
ax1.bar(range(len(cars_price.index)),cars_price)
ax1.set_xlabel("品牌",fontsize=18)
ax1.set_ylabel('价格',fontsize=18,labelpad =12)
plt.xticks(range(len(cars_price.index)),cars_price.index.values,rotation=90)
ax1.set_title('不同品牌汽车平均价格')
# 设置位置
ax2 = fig.add_subplot(222)
ax2.bar(range(len(taobao_total.index)),taobao_total)
ax2.set_xlabel('位置',fontsize =15)
ax2.set_ylabel('总价',fontsize =15)
plt.xticks(range(len(taobao_total.index)),taobao_total.index.values)
ax2.set_title('不同位置的成交总价')
# 设置位置

ax3 = fig.add_subplot(224)
ax3.bar(range(len(taobao_qty.index)),taobao_qty)
ax3.set_xlabel('位置',fontsize =15)
ax3.set_ylabel('总量',fontsize =15)
ax3.set_xticks(range(len(taobao_qty.index)))  # 位置
ax3.set_xticklabels(taobao_qty.index.values)
ax3.set_title('不同位置的成交总量')
plt.show()


#等分画布
fig = plt.figure(figsize=(16,9))
plt.subplot2grid((2,2),(0,0),rowspan=2) #设置绘图区域
plt.bar(range(len(cars_price.index)),cars_price)
plt.xlabel("品牌",fontsize=18)
plt.ylabel('价格',fontsize=18)
plt.xticks(range(len(cars_price.index)),cars_price.index.values,rotation=90)
plt.title('不同品牌汽车平均价格')

plt.subplot2grid((2,2),(0,1)) #设置绘图区域
plt.bar(range(len(taobao_total.index)),taobao_total)
plt.xlabel('位置',fontsize =15)
plt.ylabel('总价',fontsize =15)
plt.xticks(range(len(taobao_total.index)),taobao_total.index.values)
plt.title('不同位置的成交总价')

plt.subplot2grid((2,2),(1,1)) #设置绘图区域
plt.bar(range(len(taobao_qty.index)),taobao_qty)
plt.xlabel('位置',fontsize =15)
plt.ylabel('总量',fontsize =15)
plt.xticks(range(len(taobao_qty.index)),taobao_qty.index.values)  # 位置
plt.title('不同位置的成交总量')
plt.show()