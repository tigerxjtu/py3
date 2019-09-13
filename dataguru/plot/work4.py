# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:11:17 2019

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# 支持中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

os.chdir(r'D:\BaiduNetdiskDownload\Python数据可视化实战\第四章')

Titanic = pd.read_csv('titanic_train.csv')
Titanic.info()
Titanic.head()
data=Titanic['Pclass'].value_counts()
data=data.sort_index()

std_err = [20,15,26]
error_attri = dict(elinewidth=2,ecolor='black',capsize=3)
colors= ['#e41a1c','#377eb8','#4daf4a']
fig =plt.figure(figsize=(8,7)) # 创建画布
plt.bar(data.index.values,data,width=0.6,align='center',yerr=std_err,error_kw=error_attri,color= colors)
plt.xticks(data.index.values)
plt.xlabel('Pclass')
plt.ylabel('人数')
plt.grid(True,axis='y',ls=':',color='gray',alpha=0.8)
plt.title('Titanic Pclass',fontsize = 25)
plt.show()


creditcard = pd.read_csv('creditcard_exp.csv')
creditcard.info()
creditcard['edu_class'].value_counts().index
group_income = creditcard.groupby('edu_class')
avg_income = group_income.aggregate({'Income':np.mean})

colors_creditcard = ['steelblue','indianred','green','blue']

fig =plt.figure(figsize=(8,7)) # 创建画布
plt.bar(avg_income.index.values,avg_income.Income.values,width=0.6,
        align='center',color= colors_creditcard)
plt.xticks(avg_income.index)
plt.xlabel('edu_class')
plt.ylabel('人均收入')
plt.grid(True,axis='y',ls=':',color='gray',alpha=0.8)
plt.title('不同edu_class对应的收入',fontsize = 25)
# 添加表格
col_labels = ['收入']
row_labels =avg_income.index
table_vals =np.array(avg_income.Income.values).reshape(-1,1)

my_table = plt.table(cellText=table_vals,cellLoc='center' ,colWidths=[2] * 4,
   rowLabels=row_labels, colLabels=col_labels,rowColours=colors_creditcard
   ,bbox=[0.1,0.7,0.1,0.25])
plt.show()

Prod_Trade = pd.read_excel('Prod_Trade.xlsx')

Prod_Trade=Prod_Trade.sort_values('Date')
fig =plt.figure(figsize=(16,12))
# 绘制销量折线图
plt.plot(Prod_Trade.Date, # x轴数据
         Prod_Trade.Sales, # y轴数据
         linestyle = ':', # 折线类型
         color = 'steelblue', # 折线颜色
         label = '销量')
import matplotlib as mpl
# 获取图的坐标信息
ax = plt.gca()
# 设置日期的显示格式
date_format = mpl.dates.DateFormatter("%Y-%m-%d")
ax.xaxis.set_major_formatter(date_format)
# 设置x轴显示多少个日期刻度
xlocator = mpl.ticker.LinearLocator(20)
# 设置x轴每个刻度的间隔天数
#xlocator = mpl.ticker.MultipleLocator(5)
ax.xaxis.set_major_locator(xlocator)
# 为了避免x轴刻度标签的紧凑，将刻度标签旋转45度
plt.xticks(rotation=45)
plt.xlabel('日期')
# 添加y轴标签
plt.ylabel('销量')
# 添加图形标题
plt.title('每天销量图')
# 添加图例
plt.legend()
# 显示图形
plt.show()




