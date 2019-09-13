# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

os.chdir(r"D:\BaiduNetdiskDownload\Python金融业数据化运营实战\第三章\作业")
df = pd.read_csv("moisture.csv")
mu = df.moisture.mean()
sigma =  df.moisture.std()
#计算区间
print(stats.t.interval(0.95,len(df)-1,mu,stats.sem(df.moisture)))


import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False

eudf  = pd.read_csv('EuStockMarkets.csv')

eudf.plot(x= 'DAX',y='FTSE',kind='scatter',title='德国DAX指数和英国FTSE指数的散点图')
plt.show()

#计算相关系数
print(eudf[['DAX','FTSE']].corr(method='pearson'))
print(eudf[['DAX','FTSE']].corr(method='spearman'))
print(eudf[['DAX','FTSE']].corr(method='kendall'))


from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

df_house  = pd.read_csv('house.csv',skipinitialspace = True)
#ana = ols('space ~ C(education) + C(unit) + C(income) + C(type)', data = df_house).fit()
ana = ols('space ~ education + unit + income + type', data = df_house).fit()
anova_lm(ana)
ana.summary()



