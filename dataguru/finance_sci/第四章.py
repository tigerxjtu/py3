# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:32:47 2018

@author: liyun
"""
import numpy as np
import matplotlib.pyplot as plt
#%%
#==============================================================================
# 4.1.1 多项式表达方式
#==============================================================================
p = np.array([1,0,-3,5])
x = 5
print(np.polyval(p,x))

x = [1,4,5]
print(np.polyval(p,x))
#%%

#==============================================================================
# 4.1.2 多项式求解
#==============================================================================
p = np.array([1,0,-3,5])
b = np.roots(p)
print(b)
print(np.real(b))
#%%
#==============================================================================
# 4.1.3多项式乘法
#==============================================================================
a =[1,2,3,4]; b = [1,4,9,16]
print(np.convolve(a,b))
#%%

#==============================================================================
# 4.2.1 函数拟合
#==============================================================================
#%%
x = [1,2,3,4,5]
y = [5.6,40,150,250,498]
p = np.polyfit(x,y,3)
print (p)
#%%
#拟合结果
x2 = np.arange(0,5.1,0.1)
y2 = np.polyval(p,x2)
plt.plot(x,y,'*',x2,y2)
plt.show()
#%%
#==============================================================================
# #4.2.2 多项式插值
#==============================================================================
import math as math
x = list(np.arange(0,2*math.pi,0.2))

def g(x):
    return 4.8*math.cos(math.pi*x/20.0)

#y = math.cos(math.pi*x/20.0)
y = []
for t in x:
    g1 = g(t)
    y.append(g1)
    
plt.plot(x,y,'*')
plt.show()


#%%
#第二种方法创建x和y
g = lambda x: 4.8*math.cos(math.pi*x/20.0)
x = list(np.arange(0,2*math.pi,0.2))
y = [g(x) for x in list(np.arange(0,2*math.pi,0.2))]
plt.plot(x,y,'*')
plt.show()
#%%
# %%
#多项式插值
import scipy.interpolate as itp
x1 = np.arange(0,2*math.pi,0.1)
g2 = itp.interp1d(x,y,kind='cubic')
y1 =g2(x1)
#作图
plt.plot(x,y,'*',x1,y1)
plt.show()

# %%
#==============================================================================
# 4.3.1数值积分计算
#==============================================================================
import scipy.integrate
import math
F = lambda x: x**3 - 2 *x -5
Q =scipy.integrate.quad(F,0,2)

print(Q[0])

t = np.arange(0,2.1,0.01)
g = F(t)

plt.plot(t,g,'*')
plt.show()
#%%
# 计算积分求解
def h(x):
    return 0.25*x**4 - x**2 -5*x

h(2) - h(0)
#%%
#==============================================================================
# 4.3.2符号积分计算
#SymPy 的核心功能是将数学符号表示为 Python 对象。
#在 SymPy 库中，类 sympy.Symbol 可用于此目的
#==============================================================================
from sympy import *
x =  Symbol("x")
y =  Symbol('y',real=True)
s = integrate(x * y, (x,int("0"), int("1")), (y,int("1"),int("2")))
print(s)
#%%
#或者
s =  integrate(x * y, (x,int("0"), int("1")))
ss = integrate(s, (y,int("1"), int("2")))
print(s)
print(ss)
#%%
#==============================================================================
# 4.3.3导数计算
#==============================================================================
#求导
x = Symbol('x',real=True)
y = Symbol('y',real=True)
z = Symbol("z")
expr = x**4 + x**3 + x**2 + x + 1
expr.diff(x)
#%%
#多元函数
from sympy import *
expr = (x + 1) ** 3 * y ** 2 *(z - 1)
expr.diff(x,y)
#%%

#==============================================================================
# 4.4.1 线性方程组求解
#==============================================================================
A = np.array([[4,-2,1],[-2,4,-2],[1,-2,4]])
B = np.array([11,-16,17])
print(np.linalg.solve(A,B))
#%%
#==============================================================================
# 4.4.2 矩阵特征值和特征向量
#==============================================================================
A = np.array([[4,-2,1],[-2,4,-2],[1,-2,4]])
d,v = np.linalg.eig(A)
print(d,v)
#%%
#验证
print(np.dot( np.matrix(A), np.matrix(v[:,1].reshape(-1,1))))
print(np.dot(np.matrix(v[:,1].reshape(-1,1)),d[1]))
#%%
#==============================================================================
# 4.4.3可逆矩阵
#==============================================================================
A = np.matrix([[4,-2,1],[-2,4,-2],[1,-2,4]])
B = np.linalg.inv(A)
C = A*B

#%%
#==============================================================================
# 4.5资本资产定价模型与证券市场线
#==============================================================================
#Python计算但资产的风险情况
import os
import pandas as pd
os.chdir('E:\Python培训(炼数成金)\课件\第四章')
indexcd = pd.read_csv("TRD_Index.csv",index_col = 'Trddt')
#获取中证流通指数的收益率
mktcd = indexcd[indexcd.Indexcd ==902]

mktret = pd.Series(mktcd.Retindex.values,index = pd.to_datetime(mktcd.index))
mktret.name= 'market'

mktret = mktret['2014-01-01':'2014']

#获取新安股份的数据
xin_an = pd.read_csv('xin_an.csv',index_col='Date')
xin_an.index = pd.to_datetime(xin_an.index)

#剔除交易量等于0的数据
xin_an = xin_an[xin_an.Volume !=0]
xin_an['return'] = (xin_an['Close'] - xin_an['Close'].shift(1))/xin_an['Close'].shift(1)
#保留收益率变量
xin_an = xin_an['return']
xin_an.dropna(inplace=True)
#%%
#将新安股份收益率和市场收益率数据进行合并，计算风险溢价
Ret = pd.merge(pd.DataFrame(mktret),pd.DataFrame(xin_an),left_index=True,right_index=True,
               how ='inner')
#计算无风险收益率
rf = 1.036**(1/365) -1
Ret['risk_premium'] = Ret['market'] - rf

#%%
#绘制新安股份和中证指数的散点图
import matplotlib.pyplot as plt
plt.scatter(Ret['return'],Ret['market'])
plt.xlabel('xin an return'); plt.ylabel('market')
plt.title('xinan return VS market return')
#%%
#拟合曲线，找到beta
#提出X和Y
import  statsmodels.api as sm 
Ret['constant'] = 1 #增加截距项
X  = Ret[['constant','risk_premium']]
Y = Ret['return']

model= sm.OLS(Y,X)
result =model.fit()
print(result.summary())
#result.predict([1,1])
#%%
#==============================================================================
# 4.7 Fama-French三因子模型
#==============================================================================
# 华夏银行案例
import os
import pandas as pd
os.chdir('E:\Python培训(炼数成金)\课件\第四章')
#%%
stock=pd.read_table('stock.txt',sep='\t',index_col='Trddt')

HXBank = stock[stock.Stkcd==600015] #获取华夏银行数据

HXBank.index=pd.to_datetime(HXBank.index)
HXRet=HXBank.Dretwd
HXRet.name='HXRet'
HXRet.plot()
#%%
#读取三因子数据
ThreeFactors=pd.read_table('ThreeFactors.txt',sep='\t',
                           index_col='TradingDate')
#将索引转化为时间格式
ThreeFactors.index=pd.to_datetime(ThreeFactors.index)
ThrFac=ThreeFactors['2014-01-02':] #截取2014年1月2号以后的数据
ThrFac=ThrFac.iloc[:,[2,4,6]] #提取对应的3个因子
#合并股票收益率和3因子的相关数据
HXThrFac=pd.merge(pd.DataFrame(HXRet),pd.DataFrame(ThrFac),
                  left_index=True,right_index=True)

#%%
#作图
import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.scatter(HXThrFac.HXRet,HXThrFac.RiskPremium2)
plt.subplot(2,2,2)
plt.scatter(HXThrFac.HXRet,HXThrFac.SMB2)
plt.subplot(2,2,3)
plt.scatter(HXThrFac.HXRet,HXThrFac.HML2)
plt.show()
#%%
#回归
import statsmodels.api as sm
regThrFac=sm.OLS(HXThrFac.HXRet,sm.add_constant(HXThrFac.iloc[:,1:4]))
result=regThrFac.fit()
result.summary()

result.params































