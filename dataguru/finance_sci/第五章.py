# 第五章代码
# 5.2 增量法
# 定义相关参数
# f --待求解函数
# a --起始点
# b --边界点
# dx --增加量
import matplotlib.pyplot as plt
import numpy as np
def incremental_search(f,a,b,dx):
    fa = f(a)
    c = a + dx
    fc = f(c)
    n = 1
    while np.sign(fa) == np.sign(fc):
        if a >= b:
            return a - dx, n
        a = c
        fa = fc
        c = a + dx
        fc = f(c)
        n+=1
    if fa ==0:
        return a,n
    elif fc ==0:
        return c,n
    else:
        return (a+c)/2,n

y = lambda x: x**3 + 2*x**2 -5*x + 9
root,iteration = incremental_search(y,-6,3,0.001)
print('根是%.9f, 迭代%s次' % (root,iteration))

x = np.arange(-6,3,0.01)
f = y(x)
ax = plt.gca()
ax.axhline(0, xmin=-5, xmax=5, linewidth=0.3, color='r')
plt.plot(x,f)
plt.show()

#5.3 二分法
def bisection(f,a,b,tol=0.01,maxiter=100):
    c = (a+b)/2
    n = 1 # first iteration
    while n <= maxiter:
        c = (a + b)/2
        if f(c) ==0 or abs(a-b)/2 <tol:
            return c,n
        n +=1
        if f(c) < 0:
            a = c
        else:
            b = c
    return c,n

#定义一个函数
def f(x):
    return  x**3 + 2*x**2 -5*x + 9

root,iteration = bisection(f,-6,3,tol=0.0000001,maxiter=1000)
print('根是%.9f, 迭代%s次' % (root,iteration))

# 5.4 牛顿迭代法
def newton(f,df,x,tol=0.001,maxiter=100):
    n =1
    while n <= maxiter:
        x1 = x - f(x)/df(x)
        if abs(x1-x) < tol:
            return x1,n
        else:
            x = x1
            n+=1
    return None,n

def f(x):
    return  x**3 + 2*x**2 -5*x + 9

def df(x):
    return 3*x**2 + 4 * x - 5

root,iteration = newton(f,df,-6,tol=0.0000001,maxiter=100)
print('根是%.9f, 迭代%s次' % (root,iteration))


# 5.5利用scipy求解
# 定义函数和导数
import scipy.optimize as optimize
def f(x):
    return  x**3 + 2*x**2 -5*x + 9

def df(x):
    return 3*x**2 + 4 * x - 5

print('二分法的结果是:%s' % optimize.bisect(f,-6,3,xtol=0.0001))
print('牛顿法结果是newton:%s' % optimize.newton(f,3,fprime=df)) #可以不用指定
print('牛顿法结果是newton:%s' % optimize.newton(f,3)) #可以不用指定

#5.6 期权案例
# 看涨期权
import  numpy as np
import matplotlib.pyplot as plt
# 定义期权payoff函数
def pay_off_call(ST,x):
    return (ST -x + abs(ST -x))/2
x = 20.0
ST = np.arange(10,80,1)
pay_off = pay_off_call(ST,x)

plt.ylim(-10,50)
plt.plot(ST,pay_off)
plt.title('pay_off for call option')
plt.show()

#看跌期权
def pay_off_call(ST,x):
    return (x -ST + abs(x -ST))/2

import  numpy as np
import matplotlib.pyplot as plt

x = 50.0
ST = np.arange(10,80,1)
pay_off = pay_off_call(ST,x)

plt.ylim(-50,50)
plt.plot(ST,pay_off)
plt.title('pay_off for put option')
plt.show()

# 绘图看涨期权交易双方收益图
# 定义相关函数
def pay_off_call(ST,x,c):
    return (ST -x + abs(ST -x))/2 -c
import scipy as sp
s = np.arange(20,80,2)
x = 45; c=5
profit = pay_off_call(s,x,c)
y2 = np.zeros(len(s))
plt.ylim(-30,50)
plt.plot(s,profit)
plt.plot(s,y2,'-.')
plt.plot(s,-profit)
plt.title('Profit/Loss function')
plt.xlabel('Stock Price')
plt.ylabel('Profit Loss')
plt.annotate('Call option buyer',xy=(55,15),xytext=(35,20), arrowprops=dict(facecolor='blue',shrink=0.01),)
plt.annotate('Call option seller',xy=(55,-10),xytext=(40,-20), arrowprops=dict(facecolor='red',shrink=0.01),)
plt.show()

#期权定价BS
# 看涨期权定价公式
from scipy import stats
import numpy as np
def bs_call(S,X,T,r,sigma):
    d1 = (np.log(S/X) + (r+0.5*pow(sigma,2))*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S*stats.norm.cdf(d1,0,1) - X*np.exp(-r*T)*stats.norm.cdf(d2,0,1)
    return call_price

print('该欧式看涨期权费用为:%.3f' % bs_call(40,42,0.5,0.015,0.2))
































