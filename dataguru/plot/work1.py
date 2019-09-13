# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.dates as mdate

os.chdir(r'C:\download\Python数据可视化实战\第一周')

data = pd.read_csv('ChinaBank.csv',index_col=0)
data.info()
data.Date=pd.to_datetime(data.Date,format='%Y-%m-%d')
data.index=data.Date
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig1 = plt.figure(figsize=(15,6))
ax1 = fig1.add_subplot(1,1,1)
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
# 绘制单条折线图
plt.plot(data.Date, # x轴数据
         data.Close, # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 1, # 折线宽度
         color = 'skyblue', # 折线颜色
         marker = 'o', # 折线图中添加圆点
         markersize = 2, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='red') # 点的填充色
# 添加y轴标签
plt.ylabel('收盘价')
plt.xticks(pd.date_range(data.index[0],data.index[-1],freq='15D'),rotation=45)
# 添加图形标题
plt.title('中国银行股票的收盘价序列图')
# 显示图形
plt.show()

data.Date.iloc(-1)[0]
data.index=data.Date

data.index[-1]

