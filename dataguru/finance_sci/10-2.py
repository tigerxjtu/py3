# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:26:48 2018

@author: DELL
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import os

import tushare as ts

# 使用均值回归策略
# 获取数据
hs_300 = ts.get_k_data('hs300',start='2016-01-01',end='2018-11-30')[['date','close']]
hs_300.rename(columns={'close': 'price'},inplace=True)

# 计算沪深300指数收益
hs_300['return'] = np.log(hs_300['price']/hs_300['price'].shift(1))
#hs_300['return'] = hs_300['price']/hs_300['price'].shift(1) - 1 #每天收益

ma_list = [5,20]

for ma in ma_list:
    hs_300['MA_' + str(ma)] = hs_300['price'].rolling(ma).mean()

hs_300['distance'] =hs_300['MA_5']  - hs_300['MA_20'] #计算5日线和20日线价差

# 确定仓位

hs_300['position'] = np.where(hs_300['distance'] >10, 1 ,np.nan)
hs_300['position'] = np.where(hs_300['distance'] <-10 ,-1,hs_300['position'])
hs_300['position'] = np.where(np.abs(hs_300['distance']) < 10, 0 ,hs_300['position'])

hs_300['position'] = hs_300['position'] .ffill()#使用前向填充，在没有发出交易信号之前，都采用之前的
hs_300['position'].fillna(0,inplace=True)

# 绘制仓位图
hs_300['position'].plot(ylim=[-1.1,1.1])
# 计算策略收益
hs_300.index = pd.to_datetime(hs_300.date)
hs_300['stratecy'] = hs_300['position'].shift(1) * hs_300['return']
hs_300[['return','stratecy']].dropna().cumsum().apply(np.exp).plot() #计算累计收益情况，大于1才说明赚钱
#hs_300[['return','stratecy']].dropna().cumsum().plot()
plt.legend(['沪深300收益','策略收益'])
plt.show()

hs_300[['return','stratecy']].dropna().cumsum().apply(np.exp).iloc[-1]