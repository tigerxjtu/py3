# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

ra=0.04+1.2*(0.1-0.04)
print(ra-0.09)

rb=0.04+1.3*(0.1-0.04)
print(rb-0.12)

import pandas_datareader.data as web
import datetime as dt
nyyh = web.DataReader('601288.SS','yahoo', dt.datetime(2014,1,1),dt.datetime(2014,12,31))

import os
import pandas as pd
os.chdir('D:\BaiduNetdiskDownload\Python金融业数据化运营实战\第四章\作业')
indexcd = pd.read_csv("TRD_Index.csv",index_col = 'Trddt')
#获取中证流通指数的收益率
mktcd = indexcd[indexcd.Indexcd ==902]

mktret = pd.Series(mktcd.Retindex.values,index = pd.to_datetime(mktcd.index))
mktret.name= 'market'

mktret = mktret['2014-01-01':'2014']

#剔除交易量等于0的数据
nyyh = nyyh[nyyh.Volume !=0]
nyyh['return'] = (nyyh['Close'] - nyyh['Close'].shift(1))/nyyh['Close'].shift(1)
#保留收益率变量
nyyh = nyyh['return']
nyyh.dropna(inplace=True)

#将新安股份收益率和市场收益率数据进行合并，计算风险溢价
Ret = pd.merge(pd.DataFrame(mktret),pd.DataFrame(nyyh),left_index=True,right_index=True,
               how ='inner')
#计算无风险收益率
rf = 1.036**(1/365) -1
Ret['risk_premium'] = Ret['market'] - rf

#绘制新安股份和中证指数的散点图
import matplotlib.pyplot as plt
plt.scatter(Ret['return'],Ret['market'])
plt.xlabel('nyyh return'); plt.ylabel('market')
plt.title('nyyh return VS market return')

#拟合曲线，找到beta
#提出X和Y
import  statsmodels.api as sm 
Ret['constant'] = 1 #增加截距项
X  = Ret[['constant','risk_premium']]
Y = Ret['return']

model= sm.OLS(Y,X)
result =model.fit()
print(result.summary())
print(result.params)

Rf=0.005
RM=0.02
SMB=0.024
HML=0.018
Ri=Rf+0.01+1.2*(RM-Rf)+0.5*SMB+0.1*HML
print(Ri)


zyhy=pd.read_table('problem21.txt',sep='\t',index_col='Date')

zyhy.index=pd.to_datetime(zyhy.index)
zyhy['return'] = (zyhy['zyhy'] - zyhy['zyhy'].shift(1))/zyhy['zyhy'].shift(1)
#保留收益率变量
zyhy = zyhy['return']
zyhy.dropna(inplace=True)
zyhy.name='zyhyRet'
zyhy.plot()

#读取三因子数据
ThreeFactors=pd.read_table('ThreeFactors.txt',sep='\t',
                           index_col='TradingDate')
#将索引转化为时间格式
ThreeFactors.index=pd.to_datetime(ThreeFactors.index)
ThrFac=ThreeFactors['2014-01-02':] #截取2014年1月2号以后的数据
ThrFac=ThrFac.iloc[:,[2,4,6]] #提取对应的3个因子
#合并股票收益率和3因子的相关数据
zyhyThrFac=pd.merge(pd.DataFrame(zyhy),pd.DataFrame(ThrFac),
                  left_index=True,right_index=True)


#作图
import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.scatter(zyhyThrFac.zyhyRet,zyhyThrFac.RiskPremium2)
plt.subplot(2,2,2)
plt.scatter(zyhyThrFac.zyhyRet,zyhyThrFac.SMB2)
plt.subplot(2,2,3)
plt.scatter(zyhyThrFac.zyhyRet,zyhyThrFac.HML2)
plt.show()

#回归
import statsmodels.api as sm
regThrFac=sm.OLS(zyhyThrFac.zyhyRet,sm.add_constant(zyhyThrFac.iloc[:,1:4]))
result=regThrFac.fit()
print(result.summary())

print(result.params)






