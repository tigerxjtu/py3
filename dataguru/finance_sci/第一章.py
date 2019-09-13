# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a chapter one.
"""

# 先导入相关库
import pandas as pd
import os
import ffn
# import matplotlib.pyplot as plt
#ffn - Financial Functions for Python
# 转换到当前目录，即存放数据的目录
#数据日期为2014年1月1日到2014年12月31日
#SAPower代表“航天动力”股票，股票代码为“600343”
#DalianRP代表“大橡塑”股票，股票代码为“600346”
# 读取数据
os.chdir("E:\Python培训(炼数成金)\课件\Python Quant Book\part 3\\018")
SAPower=pd.read_csv('SAPower.csv',index_col='Date')
SAPower.index=pd.to_datetime(SAPower.index) #索引进行时间转化(时间格式)
DalianRP=pd.read_csv('DalianRP.csv',index_col='Date')
DalianRP.index=pd.to_datetime(DalianRP.index)

returnS=ffn.to_returns(SAPower.Close).dropna()
returnD=ffn.to_returns(DalianRP.Close).dropna()
print(returnS.std())
print(returnD.std())


#计算下行风险
def cal_half_dev(returns):
    mu=returns.mean() #计算均值
    temp=returns[returns<mu] #筛选出小于均值的收益
    half_deviation=(sum((mu-temp)**2)/len(returns))**0.5 #计算风险
    return half_deviation

print(cal_half_dev(returnS))
print(cal_half_dev(returnD))

#计算VaR
#历史模拟法
returnS.quantile(0.05)
returnD.quantile(0.05)

#协方差矩阵法
from scipy.stats import norm
norm.ppf(0.05,returnS.mean(),returnS.std())
norm.ppf(0.05,returnD.mean(),returnD.std())

#计算最大回测
ffn.calc_max_drawdown((1+returnS).cumprod())
ffn.calc_max_drawdown((1+returnD).cumprod())
print ("600343 股票最大回撤是: %.4f" % ffn.calc_max_drawdown((1+returnS).cumprod()))
#print ("600343 股票最大回撤是: %s" % format(ffn.calc_max_drawdown((1+returnS).cumprod()),'.4%'))

#第二种方法
import pandas as pd
import numpy as np
data=pd.read_csv('SAPower.csv',index_col='Date')
 #计算日收益率(G3-G2)/G2
data['return']=(data['Close'].shift(-1)-data['Close'])/data['Close']

#data['Close'].plot()
#计算累积收益率cumret=(1+return).cumsum
data['cumret']=np.cumprod(1+data['return'])
#fig = plt.figure()
#data['cumret'].plot()
#计算累积最大收益率,最大回撤，累积最长回撤时间
Max_cumret=np.zeros(len(data))
retracement=np.zeros(len(data))
Re_date=np.zeros(len(data))

for i in range(len(data)):
     #计算累积最大收益率
    if i==0:
        Max_cumret[0]=data['cumret'][0]
        retracement[0]=(Max_cumret[0])/(data['cumret'][0])-1
    else:
         #计算累积最大收益率
        Max_cumret[i]=max(Max_cumret[i-1],data['cumret'][i])
         #计算策略回撤
        #retracement[i]=  float((Max_cumret[i])/(data['cumret'][i])-1)
        retracement[i] = float((data['cumret'][i]) / (Max_cumret[i]) - 1)
     #计算最大回撤时间
    if retracement[i]==0:
        Re_date[i]=0
    else:
        Re_date[i]=Re_date[i-1]+1
 #计算最最大回撤幅度
retracement=np.nan_to_num(retracement)
Max_re=retracement.min()
 #计算最大回撤时间Max_reDate=Re_date.max()
print ("600343 股票最大回撤是: %.4f" % Max_re)
 
#计算最大回撤（第三种方法)
price = data['Close'].values
index_j = np.argmax(np.maximum.accumulate(price) - price)  # 结束位置
print(index_j)
index_i = np.argmax(price[:index_j])  # 开始位置，找到价格最高的时候对应的位置
print(index_i)
d = abs(price[index_j] - price[index_i]) # 最大回撤的值
#print(d/price[index_i]) #最大回撤比例
print ("600343 股票最大回撤是: %.4f" % (d/price[index_i]))

