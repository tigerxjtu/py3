# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:46:30 2018

@author: DELL
"""

#Q1. 在课件中的案例，假设
#T=0.5 #时间
#sigma = 0.2 #股票价格的波动率
#n_steps =100 #步长
#n_simulation =10000 #模拟多少次
#S0 = 40 #股票价格初始值
#X = 40 #strike price
#现在收益率服从一个均值为 N(0.05,0.15)的正态分布，请为期权进行定价。

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy as sp

r = 0.05 #无风险利率
T=0.5 #时间
sigma = 0.15 #股票价格的波动率
n_steps =100 #步长
# sp.random.seed(10000)
n_simulation =10000 #模拟多少次
dt =T/n_steps #间隔时间
call = [] # 计算 call的值
x = range(0,n_steps,1) #循环,模拟路径长度
S = sp.zeros([n_steps], dtype=float)
S0 = 40 #股票价格初始值
X = 40 #strike price
for j in range(0,n_simulation):
    S[0] = S0
    for i in x[:-1]:
        e = sp.random.normal()
        S[i + 1] = S[i] * np.exp((r - 0.5 * pow(sigma, 2)) * dt + sigma * np.sqrt(dt) * e)
    call.append(max(S[i+1] - X,0))
# 计算均值
call_price = sp.mean(call)*np.exp(-r*T)
print('期权价格：',call_price)

#Q2. 假设以下条件
#S0 = 50
#X = 52 
#r = 0.1
#sigma = 0.4 
#T = 5/12
#n_simulation =1000
#steps = 1000
#请为欧式看跌期权定价


def monte_call_price(S0,X,r,sigma,T,n_simulation,steps):
    dt = T/steps
    call =[]
    for j in range(0, n_simulation):
        stockprice = S0
        for i in range(0,steps,1):
            e = sp.random.normal()
            stockprice *= np.exp((r - 0.5 * pow(sigma, 2)) * dt + sigma * np.sqrt(dt) * e)
        call.append(max(stockprice - X,0))

    call_price = sp.mean(call) * np.exp(-r * T)
    return call_price


S0 = 50
X = 52
r = 0.1
sigma = 0.4
T = 5/12
n_simulation =1000
steps = 1000
call_price = monte_call_price(S0,X,r,sigma,T,n_simulation,steps)
print('期权价格：',call_price)

#Q3. 按照Q2的条件，尝试改变r,T,和sigma，分别绘制欧式看涨期权和三个参数的关系？
rs=np.arange(0.05,0.16,0.01)
price=[monte_call_price(S0,X,r,sigma,T,n_simulation,steps) for r in rs]
plt.plot(rs,price)
plt.xlabel('r')
plt.ylabel('price')
plt.show()

Ts=np.arange(3/12,1,1/12)
price=[monte_call_price(S0,X,r,sigma,T,n_simulation,steps) for T in Ts]
plt.plot(Ts,price)
plt.xlabel('T')
plt.ylabel('price')
plt.show()

sigmas=np.arange(0.2,0.6,0.05)
price=[monte_call_price(S0,X,r,sigma,T,n_simulation,steps) for sigma in sigmas]
plt.plot(sigmas,price)
plt.xlabel('sigma')
plt.ylabel('price')
plt.show()

