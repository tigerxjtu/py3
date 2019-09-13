# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:52:38 2018

@author: liyun
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
#%%
# 3.1 基本统计学概念
#==============================================================================
# 3.1.2 区间估计
#==============================================================================
#以北京住宅价格增长率为案例
#读取数据
os.chdir("E:\Python培训(炼数成金)\课件\第三章")
house_price_gr = pd.read_csv("house_price_gr.csv",encoding= 'gbk')
# 进行点估计
np.mean(house_price_gr.rate)
stats.sem(house_price_gr.rate) #样本均值的标准误
# 进行区间估计
se = stats.sem(house_price_gr.rate)
LB = house_price_gr.rate.mean()  - 1.98*se
UB = house_price_gr.rate.mean()  + 1.98*se
print (LB,UB)


#stats.t.ppf(1 - alpha/2,df)

#%%
#==============================================================================
# #上证指数的收益率
#==============================================================================
#读取数据
SHindex = pd.read_csv("TRD_Index.csv")
mu = SHindex.Retindex.mean()
sigma =  SHindex.Retindex.std()
#计算区间
stats.t.interval(0.95,len(SHindex)-1,mu,stats.sem(SHindex.Retindex))
#%%
#==============================================================================
# 3.2  假设检验与单样本t检验
#==============================================================================
import statsmodels.api as sm
d1 = sm.stats.DescrStatsW(house_price_gr.rate)
#假设为0.1
print('t检验= %6.4f,p-value=%6.4f, df=%s' % d1.ttest_mean(0.10))

#%%
#==============================================================================
# #3.3 双样本t检验
#==============================================================================
#读取数据
# 研究信用卡消费和性别的关系
creditcard_exp  = pd.read_csv('creditcard_exp.csv',skipinitialspace = True)
creditcard_exp  = creditcard_exp.dropna(how = 'any')
creditcard_exp['avg_exp'].groupby(creditcard_exp['gender']).describe().T


#先进性方差齐性检验
gender0 = creditcard_exp[creditcard_exp['gender'] ==0]['avg_exp']
gender1 = creditcard_exp[creditcard_exp['gender'] ==1]['avg_exp']
leveneTestRes = stats.levene(gender0,gender1,center='median')
print('w-value=%6.4f,p-value=%6.4f' % leveneTestRes)

#进行双样本t检验
stats.stats.ttest_ind(gender0, gender1,equal_var=True)

#%%
#==============================================================================
# #3.4 方差分析
#==============================================================================
#单因素方差分析
# 研究不同行业股票收益率水平
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
year_return = pd.read_csv('TRD_Year.csv',encoding= 'gbk')
model = ols('Return ~Industry',data =year_return.dropna()).fit()
print(anova_lm(model))

#%%
# 多因素方差分析

creditcard_exp  = pd.read_csv('creditcard_exp.csv',skipinitialspace = True)
ana = ols('avg_exp ~ C(edu_class) + C (gender)', data = creditcard_exp).fit()
anova_lm(ana)
ana.summary()
#添加交互项

ana1 = ols('avg_exp ~ C(edu_class) + C (gender) + C(edu_class)*C(gender)', data = creditcard_exp).fit()
anova_lm(ana1)
ana1.summary()
#%%
#==============================================================================
# #3.5 相关分析
#==============================================================================

creditcard_exp  = pd.read_csv('creditcard_exp.csv',skipinitialspace = True)
#画图
import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False

creditcard_exp.plot(x= 'Income',y='avg_exp',kind='scatter',title='收入与信用卡支出散点图')
plt.show()
#%%
#计算相关系数
print(creditcard_exp[['Income','avg_exp']].corr(method='pearson'))
print(creditcard_exp[['Income','avg_exp']].corr(method='spearman'))
print(creditcard_exp[['Income','avg_exp']].corr(method='kendall'))
#%%

#画散点图
import seaborn as sns
creditcard_exp1 = creditcard_exp.dropna()
sns.pairplot(creditcard_exp1[['Income','avg_exp','Age','dist_home_val','dist_avg_income']],size =2.5)
plt.show()
#%%
#==============================================================================
# 3.6 卡方检验
#==============================================================================
#列联表分析
#读取数据
accepts = pd.read_csv('accepts.csv')
cross_table = pd.crosstab(accepts.bankruptcy_ind,columns=accepts.bad_ind,margins=True)
cross_table_rowpct = cross_table.div(cross_table['All'],axis=0)

#卡方检验
stats.chi2_contingency(cross_table)
#%%











