# 第七章代码
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy as sp
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#==============================================================================
# 7.2 随机数模拟
#==============================================================================
# # # # # #产生一个随机数
npr.rand(100)
npr.rand(5,5)
# 产生5到10范围内的随机数
npr.rand(100)*5 + 5

# # # # # # 模拟正态分布
# 产生一个均值为100,标准差为20的正态分布
x = npr.normal(100,20,(10000,3))
plt.hist(x,bins=30)
plt.xlabel('正态分布随机数')
plt.ylabel('频数')
plt.legend(['第一列','第二列','第三列'],loc =1)
plt.show()

# # # # # # 模拟二项分布
# 发生概率为0.3，相当于在10次实验中，时事件发生的次数
# npr.seed(100) #随机数种子一定时，产生的随机数相同
import seaborn as sns
# npr.seed(100)
x = npr.binomial(n=10,p=0.3,size=10000)
sns.distplot(x,bins=30,kde=False)
plt.xlabel('发生次数的随机数分布')
plt.ylabel('频数')
plt.show()


# # # # # # 模拟指数分布
x = npr.exponential(scale=1,size=100)
plt.hist(x,bins=10)
plt.xlabel('指数分布随机数分布')
plt.ylabel('频数')
plt.show()

# 概率密度函数
lambd = 1
x = np.arange(0,15,0.1)
y = lambd*np.exp(-lambd*x)
plt.plot(x,y)
plt.show()

# 将密度函数和模拟结果画在一起
x = npr.exponential(scale=1,size=100)
lambd = 1
x1 = np.arange(0,15,0.1)
y1 = lambd*np.exp(-lambd*x1)
plt.hist(x,bins=10,label='模拟结果',density=True)
plt.plot(x1,y1,label ='指数分布')
plt.xlabel('指数分布随机数分布')
plt.ylabel('频数')
plt.legend(loc='upper right')
plt.show()


# # # # # # 模拟泊松分布
x = npr.poisson(lam=2,size=1000)
plt.hist(x,bins=10,density=True )
plt.xlabel('泊松分布随机数分布')
plt.ylabel('频数')
plt.show()

# 概率密度函数
import math
lambd = 2
x = np.arange(1,15,1)
y = []
for t in x:
    y.append((pow(lambd,t)/math.factorial(t))* np.exp(-lambd))
plt.plot(x,y)
plt.show()

# 将密度函数和模拟结果画在一起
x_moni = npr.poisson(lam=2,size=1000)
lambd = 2
x = np.arange(1,15,1)
y = []
for t in x:
    y.append((pow(lambd, t) / math.factorial(t)) * np.exp(-lambd))
plt.hist(x_moni,bins=10,label='模拟结果',density=True)
plt.plot(x,y,label ='泊松分布')
plt.xlabel('泊松分布随机数分布')
plt.ylabel('频数')
plt.legend(loc='upper right')
plt.show()

#==============================================================================
# 7.3 蒙特卡洛模拟
#==============================================================================
# 测试点数
Testnum =10000000
# 落到圆内的点数
# 产生均值分布随机数，落在[-1,1]之间
import scipy as sp
sp.random.seed(123345)
x = sp.random.uniform(low=-1,high=1,size=Testnum)
y = sp.random.uniform(low=-1,high=1,size=Testnum)
# 找出x^2 + y^2 <=1
circle_num = 0
for i in range(0,Testnum):
    if (x[i]**2 + y[i]**2) <=1:
        circle_num += 1
print('pi的值等于%.10f' % (4*circle_num/Testnum))


# # # 计算不同的模拟次数Pi的值
def calculate_pi(start_num,Testnum,steps):
    cal_pi =[]
    # 循环次数
    sim_num = np.arange(start_num,Testnum,steps)
    for num in sim_num:
        circle_num = 0
        x = sp.random.uniform(low=-1, high=1, size=num)
        y = sp.random.uniform(low=-1, high=1, size=num)
        for i in range(0,num):
            if (x[i] ** 2 + y[i] ** 2) <= 1:
                circle_num += 1
        cal_pi.append(4*circle_num/num)
    return cal_pi

sim_num = np.arange(10000, 1000000, 10000) # 模拟次数
cal_pi = calculate_pi(10000, 1000000, 10000) # Pi的值
plt.plot(sim_num,cal_pi)
plt.xlabel('次数')
plt.ylabel('圆周率的值')
plt.show()
#==============================================================================
# 7.4 价格序列
#==============================================================================
# # # # # #生成收益率服从正态分布的价格序列
def RandnPrice(Price0,mu,sigma,N):
    # 生成随机收益率
    rate  = np.random.normal(mu,sigma,N)
    #使用累积函数
    Price = Price0 * np.cumprod(rate+1)
    return Price

# 定义初始值
Price0 = 10
mu = 1.1**(1/250) - 1 #假设收益率为10%
sigma = 0.3/np.sqrt(250) #标准差
N =250 *2

Price = RandnPrice(Price0,mu,sigma,N)
plt.plot(Price)
plt.xlabel('time')
plt.ylabel('价格')
plt.show()

# # # # # # 模拟股票走势
import pandas as pd
stock_price_today = 10
T = 1 # 时间为1年
n_steps =1000 #步长
mu =0.15
sigma =0.2
n_simulation = 100 #模拟次数
dt =T/n_steps # 时间间隔
# S = sp.zeros([n_steps],dtype=float)
stock_price_path = pd.DataFrame(columns = range(0,n_simulation,1))
S = sp.zeros([n_steps], dtype=float)
x = range(0,n_steps,1)
for j in range(0,n_simulation):
    #S = sp.zeros([n_steps], dtype=float)
    S[0] = stock_price_today
    for i in x[:-1]:
        e = sp.random.normal()
        # S[i+1] = S[i]+S[i]*(mu -0.5*pow(sigma,2))*dt + sigma*S[i]*sp.sqrt(dt)*e
        S[i+1] = S[i]*np.exp((mu -0.5*pow(sigma,2))*dt + sigma*np.sqrt(dt)*e )
    stock_price_path[j] = S

plt.plot(stock_price_path)
plt.xlabel('Total_number_of_steps')
plt.ylabel('stock_price')
plt.show()
#绘制股票到期日价格分布

#==============================================================================
# 7.5 期权定价
#==============================================================================
# # # # # #   期权定价
r = 0.05 #无风险利率
T=0.5 #时间
sigma = 0.2 #股票价格的波动率
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
print('看涨期权价格为: %.2f' % call_price )

#
def monte_call_price(S0,X,r,mu,T,n_simulation,steps):
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
mu = 0.4
T = 5/12
n_simulation =1000
steps = 1000
call_price = monte_call_price(S0,X,r,mu,T,n_simulation,steps)

