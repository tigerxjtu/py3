# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy import stats
#1
def bs_call(S,X,T,r,sigma):
    d1 = (np.log(S/X) + (r+0.5*pow(sigma,2))*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S*stats.norm.cdf(d1,0,1) - X*np.exp(-r*T)*stats.norm.cdf(d2,0,1)
    return call_price

s =40
x =40
t = 0.25
r=0.05
sigma=0.20

print('该欧式看涨期权费用为:%.3f' % bs_call(s,x,t,r,sigma))
S=np.array([s for s in (range(30,60))])
Y=bs_call(S,x,t,r,sigma)
import matplotlib.pyplot as plt

plt.xlabel('Stock Price')
plt.ylabel('Call Price')
plt.plot(S,Y)
plt.show()

#2
def bs_put(S,X,T,r,sigma):
    d1 = (np.log(S/X) + (r+0.5*pow(sigma,2))*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put_price=X*np.exp(-r*T)*stats.norm.cdf(-d2,0,1)-S*stats.norm.cdf(-d1,0,1)*d2
    return put_price


s =38.5
x =37
t = 0.5
r=0.032
sigma=0.25
print('该欧式看涨期权费用为:%.3f' % bs_call(s,x,t,r,sigma))
print('该欧式看跌期权费用为:%.3f' % bs_put(s,x,t,r,sigma))


#看跌期权
def pay_off_put(ST,x,p):
    return (x-ST + abs(x-ST))/2 -p
import scipy as sp
s = np.arange(0,80,1)
x=45
p=2
profit = pay_off_put(s,x,c)
y2 = np.zeros(len(s))
plt.ylim(-50,50)
plt.plot(s,profit)
plt.plot(s,y2,'-.')
plt.plot(s,-profit)
plt.title('Profit/Loss function')
plt.xlabel('Stock Price')
plt.ylabel('Profit Loss')
plt.annotate('Put option buyer',xy=(55,-10),xytext=(40,-20), arrowprops=dict(facecolor='blue',shrink=0.01),)
plt.annotate('Put option seller',xy=(55,15),xytext=(35,20), arrowprops=dict(facecolor='red',shrink=0.01),)
plt.show()




