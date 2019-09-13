"""
这是第十章知识点
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import os

import tushare as ts
#==============================================================================
# 10.2 绘制K线图
#==============================================================================
#获取上证指数2015年3月份到12月份的看K线图数据
sh_data = ts.get_k_data('sh',start='2015-03-01',end='2015-12-31')
sh_data.reset_index(inplace=True)
from  mpl_finance import candlestick2_ohlc
fig, ax = plt.subplots()
candlestick2_ohlc(ax,sh_data['open'],sh_data['high'],sh_data['low'],sh_data['close'],\
                  colorup='red', colordown='green',width =0.5)
plt.xticks(range(0,len(sh_data.index),10),sh_data.loc[range(0,len(sh_data.index),10),'date'],\
                                           rotation=45) #时间间隔
ax.set_xlabel('日期')
ax.set_title('上证指数K线图')
ax.xaxis.set_label_coords(1.0,-0.05)# axes 坐标移动位置
plt.show()

#==============================================================================
# 10.3 移动均线
#==============================================================================
# 获取沪深300指数
hs_data = ts.get_k_data('hs300',start='2015-01-01',end='2015-12-31')
# 计算5日移动均线,20日和60日移动均线
ma_list = [5,20,60]
for ma in ma_list:
    hs_data['MA_' + str(ma)] = hs_data['close'].rolling(ma).mean()
# 绘制三个移动均线的变化趋势
fig,ax = plt.subplots()
plt.plot(hs_data['close'],linestyle=':',color='red',label='收盘价')
plt.plot(hs_data['MA_5'],linestyle='--',color='blue',label='5日均线')
plt.plot(hs_data['MA_20'],linestyle='-',color='orange',label='20日均线')
plt.plot(hs_data['MA_60'],linestyle='-',color='steelblue',label='60日均线')
plt.xticks(range(0,len(hs_data.index),10),hs_data.loc[range(0,len(hs_data.index),10),'date'],\
                                           rotation=45) #时间间隔
ax.set_xlabel('日期')
ax.set_title('上证指数移动均线图')
ax.xaxis.set_label_coords(1.0,-0.05)# axes 坐标移动位置
plt.legend(loc='upper left')
plt.show()

# 计算MACD指标
#定义函数，获取macd,导入数据，初始化三个参数
def get_macd_data(data,short=12,long=26,mid=9):
#定义相关参数
#计算短期的ema，使用pandas的ewm得到指数加权的方法，mean方法指定数据用于平均
    data['sema']=pd.Series(data['close']).ewm(span=short).mean()
#计算长期的ema，方式同上
    data['lema']=pd.Series(data['close']).ewm(span=long).mean()
#计算dif，加入新列data_dif
    data['DIF']=data['sema']-data['lema']
#计算dea
    data['DEA']=pd.Series(data['DIF']).ewm(span=mid).mean()
#计算macd
    data['MACD']=2*(data['DIF']-data['DEA'])
#返回data的三个新列
    return data

# 计算MACD
data_hs = get_macd_data(hs_data,short=12,long=26,mid=9)
# 绘制图形
fig,ax1 = plt.subplots()
#plt.plot(data_hs['close'],ls='-',color='red',label='收盘价')
plt.plot(data_hs['DIF'],ls='-',color='blue',label='DIF')
plt.plot(data_hs['DEA'],ls=':',color='green',label='DEA')
# plt.plot(data_hs['MACD'],ls='-.',color='green',label='MACD')
plt.bar(x= hs_data.loc[:,'date'],height=data_hs['MACD'],color='r',label='MACD')
ax1.set_xlabel('日期')
ax1.set_title('上证指数K线图')
ax1.xaxis.set_label_coords(1.0,-0.05)# axes 坐标移动位置
plt.xticks(range(0,len(data_hs.index),5),data_hs.loc[range(0,len(data_hs.index),5),'date'],\
                                           rotation=45) #时间间隔
ax1.set_xlabel('日期')
plt.legend(loc='upper left')
ax2 = ax1.twinx()  # this is the important function #添加次坐标轴
ax2.plot(data_hs['close'],c= 'orange',ls='-',label='收盘价')
ax2.set_ylabel('收盘价')
plt.legend(loc='upper right')
plt.show()

#==============================================================================
# 10.4 布林通道
#==============================================================================
# 以2倍标准差
# 计算20日均线 和20日均线对应的标准差
hs_data = ts.get_k_data('hs300',start='2015-01-01',end='2015-12-31')
hs_data['MA_20']  =  hs_data['close'].rolling(20).mean() #20日均线
hs_data['MA_std'] =  hs_data['close'].rolling(20).std()  #20日均线对应的标准差
hs_data['Upper'] = hs_data['MA_20']  + 2*hs_data['MA_std'] #阻力线
hs_data['Lower'] = hs_data['MA_20']  - 2*hs_data['MA_std'] #支撑线
# 绘制图形

fig,ax = plt.subplots()
plt.plot(hs_data['close'],linestyle='-',color='red',label='收盘价')
plt.plot(hs_data['MA_20'],linestyle='--',color='blue',label='20日均线')
plt.plot(hs_data['Upper'],linestyle='-.',color='orange',label='阻力线')
plt.plot(hs_data['Lower'],linestyle='-',color='steelblue',label='支撑线')
plt.xticks(range(0,len(hs_data.index),10),hs_data.loc[range(0,len(hs_data.index),10),'date'],\
                                           rotation=45) #时间间隔
ax.set_xlabel('日期')
ax.set_title('上证指数布林带图')
ax.xaxis.set_label_coords(1.0,-0.05)# axes 坐标移动位置
plt.legend(loc='upper left')
plt.show()


#==============================================================================
# 10.5 RSI相对强弱指标
#==============================================================================
# 定义一个函数用于计算RSI
def RSI(t, periods=10):
    length = len(t)
    rsies = [np.nan]*length
    #数据长度不超过周期，无法计算；
    if length <= periods:
        return rsies
    #用于快速计算；
    up_avg = 0
    down_avg = 0

    #首先计算第一个RSI，用前periods+1个数据，构成periods个价差序列;
    first_t = t[:periods+1] #python右边是闭区间
    for i in range(1, len(first_t)):
        #价格上涨;
        if first_t[i] >= first_t[i-1]:
            up_avg += first_t[i] - first_t[i-1]
        #价格下跌;
        else:
            down_avg += first_t[i-1] - first_t[i]
    up_avg = up_avg / periods
    down_avg = down_avg / periods
    rs = up_avg / down_avg
    rsies[periods] = 100 - 100/(1+rs)
    #后面的将使用快速计算；
    for j in range(periods+1, length):
        up = 0
        down = 0
        if t[j] >= t[j-1]:
            up = t[j] - t[j-1]
            down = 0
        else:
            up = 0
            down = t[j-1] - t[j]
        #类似移动平均的计算公式;
        up_avg = (up_avg*(periods - 1) + up)/periods
        down_avg = (down_avg*(periods - 1) + down)/periods
        rs = up_avg/down_avg
        rsies[j] = 100 - 100/(1+rs)
    return rsies
# 获取沪深300指数
hs_data = ts.get_k_data('hs300', start='2015-01-01', end='2015-12-31')
rsies =  RSI(hs_data.close.values, periods=10)
hs_data['RSI'] = rsies

# 绘制RSI和收盘价关系
# 绘制图形
fig,ax1 = plt.subplots()
plt.plot(hs_data['RSI'].shift(1),ls='-',color='green',label='RSI')
#plt.plot(hs_data['RSI'].shift(1),ls='-',color='green',label='RSI')
ax1.set_xlabel('日期')
ax1.set_title('RSI指标与上证指数关系')
ax1.set_ylabel('RSI指标')
ax1.xaxis.set_label_coords(1.0,-0.05)# axes 坐标移动位置
plt.xticks(range(0,len(hs_data.index),5),hs_data.loc[range(0,len(hs_data.index),5),'date'],\
                                           rotation=45) #时间间隔
ax1.set_xlabel('日期')
plt.legend(loc='upper left')
ax2 = ax1.twinx()  # this is the important function #添加次坐标轴
ax2.plot(hs_data['close'],c= 'orange',ls='-',label='收盘价')
ax2.set_ylabel('收盘价')
plt.legend(loc='upper right')
plt.show()

# 绘制超买和超卖
hs_data['RSI'].dropna().plot()
plt.axhline(80,color='r')
plt.axhline(50,color='r')
plt.axhline(20,color='r')
plt.show()

#==============================================================================
# 10.6 策略模拟
#==============================================================================
# 使用均值回归策略
# 获取数据
hs_300 = ts.get_k_data('hs300',start='2014-01-01',end='2015-11-30')[['date','close']]
hs_300.rename(columns={'close': 'price'},inplace=True)
fig,ax =plt.subplots()
plt.plot(hs_300['price'],linestyle='-',color='red',label='收盘价')
ax.set_xlabel('日期')
ax.set_title('沪深300指数')
ax.xaxis.set_label_coords(1.0,-0.05)# axes 坐标移动位置
plt.xticks(range(0,len(hs_300.index),20),hs_300.loc[range(0,len(hs_300.index),20),'date'],\
                                           rotation=45) #时间间隔
ax.xaxis.set_label_coords(1.0,-0.05)# axes 坐标移动位置
plt.show()

# 计算沪深300指数收益
hs_300['return'] = np.log(hs_300['price']/hs_300['price'].shift(1))
#hs_300['return'] = hs_300['price']/hs_300['price'].shift(1) - 1 #每天收益
SMA = 30
# 计算30日移动均值
hs_300['SMA_30'] = hs_300['price'].rolling(SMA).mean()
hs_300['distance'] =hs_300['price']  - hs_300['SMA_30'] #计算价差
# 计算价差均值
threshold = hs_300['distance'].median()
#定义一个阈值
threshold =100
# 当价差超过100时，就做空，价差小于-100时，就做多
hs_300['distance'].dropna().plot()
plt.axhline(threshold,color='r')
plt.axhline(-threshold,color='r')
plt.axhline(0,color='r')
plt.show()

# 确定仓位
hs_300['position'] = np.where(hs_300['distance'] > threshold,-1,np.nan)
hs_300['position'] = np.where(hs_300['distance'] <  -threshold,1,hs_300['position'])
hs_300['position'] = np.where(hs_300['distance']*hs_300['distance'].shift(1) < 0, 0 ,hs_300['position'])
hs_300['position'] = hs_300['position'] .ffill()#使用前向填充，在没有发出交易信号之前，都采用之前的
hs_300['position'].fillna(0,inplace=True)

# 绘制仓位图
hs_300['position'].plot(ylim=[-1.1,1.1])
# 计算策略收益
hs_300.index = pd.to_datetime(hs_300.date)
hs_300['stratecy'] = hs_300['position'].shift(1) * hs_300['return']
hs_300[['return','stratecy']].dropna().cumsum().apply(np.exp).plot() #计算累计收益情况，大于1才说明赚钱
#hs_300[['return','stratecy']].dropna().cumsum().plot()
plt.legend(['沪深300收益','策略收益'])
plt.show()



