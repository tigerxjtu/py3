# 第八章代码
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
# 支持中文显示
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_column', 8 )
#==============================================================================
# 8.2 马科维兹投资组合理论简单案例
#==============================================================================
# # # # # # 简单案例
returns = np.array([0.000540, 0.000275, 0.000236]).reshape(-1,1) #收益率(列向量)
covariance = 0.0001* np.array([[5.27, 2.80, 1.74],
                       [2.80, 4.26, 1.67],
                       [1.74, 1.67, 2.90]]) #协方差矩阵
weights = 1.0/3 * np.array([1,1,1]).reshape(-1,1) #权重矩阵(列向量)
port_expected_return = np.dot(weights.T,returns)[0] #这里是投资组合收益情况
#这里是投资组合风险情况
port_expected_risk  = np.dot(np.dot(weights.T,covariance),weights)[0] #方差
port_expected_std = port_expected_risk[0] ** 0.5 #标准差
print('投资组合收益和标准差分别为: %6e, %.4f' %(port_expected_return ,port_expected_std ) )
#==============================================================================
# 8.3 python案例实践
#==============================================================================
# # # # # # #  基于股票的历史收益来进行投资组合优化
# 产生三个数据，基于模拟法，模拟三只股票的收益率数据
stock1 = npr.normal(0.000525,0.03,100)# 第一只股票收益率
stock2 = npr.normal(-0.00044,0.03,100)# 第二只股票收益率
stock3= npr.normal(0.004,0.03,100) # 第三只股票收益率
stock_data = pd.DataFrame({'stock1': stock1,'stock2':stock2,'stock3':stock3})
selected = ['stock1','stock2','stock3']
# 用蒙特卡洛法产生大量的模拟
port_returns = [] #投资组合收益率
port_volatility = [] #波动
stock_weights = [] #权重
num_assets = 3 #资产数量
num_portfolios = 10000  #产生10000次随机模拟
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets) #随机产生一次
    weights /= np.sum(weights) #计算权重
    returns = np.dot(weights, stock_data.mean()) #期望收益
    volatility = np.sqrt(np.dot(np.dot(weights,stock_data.cov()),weights.reshape(-1,1))[0])#波动
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)
#创建一个字典，存储相关数据
portfolio = {'Returns': port_returns, 'Volatility': port_volatility}

# and weight in the portfolio 投资组合权重
for counter,symbol in enumerate(selected):
    portfolio[symbol+'_weight'] = [weight[counter] for weight in stock_weights]
df = pd.DataFrame(portfolio) #转换为dataframe

column_order = ['Returns', 'Volatility'] + [stock+'_weight' for stock in selected]
df = df[column_order]
df.head()

# 绘制图形
plt.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.scatterplot(x = 'Volatility',y = 'Returns',data = df,color="steelblue", marker='o', s=20)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('投资组合风险和收益情况')
plt.show()

# 计算夏普比最大的投资组合
# 计算夏普比
# 假设无风险收益率每天为0.04/252
df['sharp_ratio'] = (df['Returns'] - 0.04/252)/df['Volatility']
sharp_ratio = df.loc[df['sharp_ratio']==df['sharp_ratio'].max(),:]#计算夏普比例最大对应的值
min_vari = df.loc[df['Volatility']==df['Volatility'].min(),:]#计算方差最小对应的值

# 使用函数求解
# port_returns = np.array(port_returns)
# port_variance = np.array(port_returns)
num =  3 #投资组合资产个数
returns = stock_data #股票投资回报率
# 定义函数，返回投资组合预期收益,标准差和夏普比例
def statistics(weights):
    weights = np.array(weights)
    port_returns = np.dot(weights.reshape(1,-1),returns.mean())
    port_variance = np.sqrt(np.dot(np.dot(weights, returns.cov()), weights.reshape(-1, 1)))
    return np.array([port_returns, port_variance, (port_returns - 0.04/252)/port_variance])

#最优化投资组合的推导是一个约束最优化问题
import scipy.optimize as sco
#最小化夏普指数的负值
def min_sharpe(weights):
    return -statistics(weights)[2]
# 约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
cons=({'type':'eq', 'fun':lambda x: np.sum(x)-1})
#我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
bnds = tuple((0,1) for x in range(num))
#优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。
opts = sco.minimize(min_sharpe, num*[1./num,], method = 'SLSQP', bounds = bnds, constraints = cons)
opts #结算结果
opts['x'].round(3) #权重
statistics(opts['x']) # 得到投资组合，分别为收益率，方差和夏普比例

# # # # # # 方差最小
def min_variance(weights):
    return statistics(weights)[1]
optv = sco.minimize(min_variance, num*[1.0/num,], method='SLSQP',bounds=bnds,
                    constraints=cons)
optv['x'] #权重
# 得到方差最小的投资组合
statistics(optv['x'])  # 得到投资组合，分别为收益率，方差和夏普比例
# 绘制波动最小和夏普比例最高在图形上
sharpe_portfolio =  statistics(opts['x'])  #计算夏普比例最大对应的值
min_variance_port = statistics(optv['x']) ##计算方差最小对应的值
sns.scatterplot(x = 'Volatility',y = 'Returns',color='steelblue',data = df,
          marker='D', s=20)
plt.scatter(x=sharp_ratio.Volatility, y=sharp_ratio.Returns, c='darkblue', marker='o', s=50) #模拟结果
plt.scatter(x=min_vari.Volatility, y=min_vari.Returns, c='orange', marker='D', s=50 )#模拟结果
plt.scatter(x=sharpe_portfolio[1], y=sharpe_portfolio[0], c='red', marker='o', s=50)#数值计算结果
plt.scatter(x=min_variance_port[1], y=min_variance_port[0], c='blue', marker='D', s=50 )#数值计算结果
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('投资组合风险和收益情况')
plt.show()


# # # # # # # 投资组合实际案例运用
# 通过真实股票数据案例
import tushare as ts
# 股票池
symbol = ['002697','600783','000413','601588']
# 002697 红旗连锁, 600783 鲁信创投, 000413 东旭光电, 601588 北辰实业
data = ts.get_k_data('hs300',start='2015-01-01',end='2015-12-31')
data = data[['date','close']]
data.rename(columns={'close': 'hs300'},inplace=True)
# 分别为沪深300,北京银行，航天动力和上海能源
# data = pd.DataFrame()
for i in symbol:
    get_data = ts.get_k_data(i,start='2015-01-01',end='2015-12-31')
    get_data = get_data[['date','close']]
    get_data.rename(columns={'close': i + '_close'},inplace=True)
    data = pd.merge(data,get_data,left_on='date',right_on='date',how='left')
data.index = data['date']
del data['date']
del data['hs300']
data = data.dropna() #删除缺失值
data.index = pd.to_datetime(data.index)
(data/data.iloc[0]*100).plot(figsize=(8,4)) #量纲级处理

# 计算收益率
returns = np.log(data/data.shift(1))
returns = returns.dropna()
# 给不同资产分配权重
# 用蒙特卡洛法产生大量的模拟
port_returns = [] #投资组合收益率
port_volatility = [] #波动
stock_weights = []#权重
num_assets =4 #资产数量
num_portfolios = 10000 #产生10000次随机模拟

for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    port_returns.append(np.dot(weights, returns.mean()*252))#期望收益
    volatility = np.sqrt(np.dot(np.dot(weights,returns.cov()*252),weights.reshape(-1,1))[0])#波动
    port_volatility.append(volatility)
    stock_weights.append(weights)

portfolio = {'Returns': port_returns, 'Volatility': port_volatility} #创建一个字典
# and weight in the portfolio 投资组合权重
for counter,stock in enumerate(symbol):
    portfolio[stock +'_weight'] = [weight[counter] for weight in stock_weights]
df = pd.DataFrame(portfolio)
#按顺序取数
column_order = ['Returns', 'Volatility'] + [stock+'_weight' for stock in symbol]
df = df[column_order]
df.head()
# 绘制图形
sns.scatterplot(x = 'Volatility',y = 'Returns',data = df,color="steelblue", marker='o', s=20)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('投资组合风险和收益情况')
plt.show()

# 计算夏普比最大的投资组合
# 计算夏普比
# 假设无风险收益率每天为0.04
df['sharp_ratio'] = (df['Returns'] - 0.04)/df['Volatility']
sharp_ratio = df.loc[df['sharp_ratio']==df['sharp_ratio'].max(),:]#计算夏普比例最大对应的值
min_vari = df.loc[df['Volatility']==df['Volatility'].min(),:]#计算方差最小对应的值
# 使用函数求解
num = 4 #投资组合资产个数
# 定义函数，返回投资组合预期收益,标准差和夏普比例
def statistics(weights):
    weights = np.array(weights)
    port_returns = np.dot(weights.reshape(1,-1),returns.mean()*252)
    port_variance = np.sqrt(np.dot(np.dot(weights, returns.cov()*252), weights.reshape(-1, 1)))
    return np.array([port_returns, port_variance, (port_returns - 0.04)/port_variance])

#最优化投资组合的推导是一个约束最优化问题
import scipy.optimize as sco
#最小化夏普指数的负值
def min_sharpe(weights):
    return -statistics(weights)[2]
# 约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
cons=({'type':'eq', 'fun':lambda x: np.sum(x)-1})
#我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
bnds = tuple((0,1) for x in range(num))
#优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。
opts = sco.minimize(min_sharpe, num*[1./num,], method = 'SLSQP', bounds = bnds, constraints = cons)
opts #结算结果
opts['x'].round(3) #权重
statistics(opts['x']) # 得到投资组合，分别为收益率，方差和夏普比例

# # # # # # 方差最小
def min_variance(weights):
    return statistics(weights)[1]

optv = sco.minimize(min_variance, num*[1.0/num,], method='SLSQP',bounds=bnds,
                    constraints=cons)
optv['x'] #权重
# 得到方差最小的投资组合
statistics(optv['x'])  # 得到投资组合，分别为收益率，方差和夏普比例

# # # 投资组合有效边界
def min_variance(weights):
    return statistics(weights)[1]

#在不同目标收益率水平（target_returns）循环时，最小化的一个约束条件会变化。
target_returns = np.linspace(0.2,0.56,50)
target_variance = []
for tar in target_returns:
    cons = ({'type':'eq','fun':lambda x:statistics(x)[0]-tar},{'type':'eq','fun':lambda x:np.sum(x)-1})
    res = sco.minimize(min_variance, num*[1./num,],method = 'SLSQP', bounds = bnds, constraints = cons)
    target_variance.append(res['fun'])
target_variance = np.array(target_variance)

# 绘制波动最小和夏普比例最高在图形上
sharpe_portfolio =  statistics(opts['x'])  #计算夏普比例最大对应的值
min_variance_port = statistics(optv['x']) ##计算方差最大对应的值
sns.scatterplot(x = 'Volatility',y = 'Returns',color='steelblue',data = df,
          marker='D', s= 20)
plt.scatter(x= sharpe_portfolio[1], y=sharpe_portfolio[0], c='red', marker='o', s=50)
plt.scatter(x= min_variance_port[1], y=min_variance_port[0], c='blue', marker='D', s=50 )
plt.scatter(x=min_vari.Volatility, y=min_vari.Returns, c='red', marker='o', s=50)
plt.scatter(x=sharp_ratio.Volatility, y=sharp_ratio.Returns, c='blue', marker='D', s=50 )
# 有效边界
#叉号：有效前沿
plt.scatter(target_variance,target_returns, marker = 'x')
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('投资组合风险和收益情况')
plt.show()

