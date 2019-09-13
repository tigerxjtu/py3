"""
这是第7章的知识点
"""
# 第七章  章 Python高级绘图三实现共享坐标轴
# 导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from  ggplot import *
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.chdir('c:\data')
pd.set_option('display.max_columns',8)
#==============================================================================
# 7.1 共享单一绘图区域的坐标轴
#==============================================================================
jd_stock  = pd.read_csv('data.csv', sep =',',header=None,names =['name','date','opening_price','closing_price',
                                                                 'lowest_price','highest_price','volume'])
# jd_stock['date'] = pd.to_datetime(jd_stock['date'])
fig, ax1 = plt.subplots() #创建画图对象
ax1.plot(jd_stock['date'],jd_stock['opening_price'],c='b',ls='--',label='开盘价')
ax1.legend(loc='upper right')
ax1.set_xticks(range(0,70,4))  # 位置
ax1.set_xticklabels(jd_stock.iloc[range(0,70,4),1],rotation=45)
ax1.xaxis.set_label_coords(1.05,-0.02)
ax1.set_xlabel('日期',fontsize=10)
ax1.set_ylabel('开盘价',color='b')
ax1.tick_params('y',colors='b') #使坐标轴的和线条相匹配
ax2 = ax1.twinx() #使用子坐标
ax2.plot(jd_stock['date'],jd_stock['volume'],c='r',ls=':',label='成交量')
ax2.set_xticks(range(0,70,4))  # 位置
ax2.set_xticklabels(jd_stock.iloc[range(0,70,4),1],rotation=45)
ax2.legend(loc='upper left')
ax2.set_xlabel('日期',fontsize=10)
ax2.set_ylabel('成交量',color='r')
ax2.tick_params('y',colors='r') #使坐标轴的和线条相匹配
plt.show()

#==============================================================================
# 7.2 共享不同区域的坐标轴
#==============================================================================
x1 = np.linspace(0,2*np.pi,400)
y1 = np.cos(x1**2)
x2 = np.linspace(0.01,10,100)
y2 = np.sin(x2)
x3 = np.random.randn(100)
y3 = np.linspace(0,3,100)
x4 = np.arange(0,6,0.5)
y4 = np.power(x4,3)
# fig,ax = plt.subplots(2,2)
fig,ax = plt.subplots(2,2,sharex = 'row') #可以取'none','row','col','all'
ax1 = ax[0,0]
ax1.plot(x1,y1)
ax2= ax[0,1]
ax2.plot(x2,y2)
ax3 = ax[1,0]
ax3.scatter(x3,y3)
ax4 =ax[1,1]
ax4.scatter(x4,y4)
plt.show()
# 根据sharex取值来判断共享那个轴,请大家自己进行参数替换，一一观察效果
# # # 将空隙去掉
x = np.linspace(0,1,200)
y = np.cos(x)*np.sin(x)
y2 = np.exp(-x)*np.sin(x)
y3 =3*np.sin(x)
y4 =np.power(x,0.5)
fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex='all',figsize=(16,9))
fig.subplots_adjust(hspace = 0) #去掉空间
ax1.plot(x,y,ls='-',lw=2,label='a')
ax1.set_yticks(np.arange(-0.6,0.7,0.2))
ax1.legend(loc='upper left')
ax1.set_ylim(-0.7,0.7)
ax2.plot(x,y2,ls='-',lw=2)
ax2.set_yticks(np.arange(-0.05,0.7,0.2))
ax2.set_ylim(-0.1,0.4)
ax3.plot(x,y3,ls='-',lw=2)
ax3.set_yticks(np.arange(-3,4,1))
ax3.set_ylim(-3.5,3.5)
ax4.plot(x,y4,ls='-',lw=2,label='d')
ax4.set_yticks(np.arange(0.0,3.6,0.5))
ax4.set_ylim(0,4.0)
ax4.legend(loc='upper left')
plt.show()

# # #  共享个别区域坐标轴
# 与第一幅图共享x
x1 = np.linspace(0,2*np.pi,400)
y1 = np.cos(x1**2)
x2 = np.linspace(0.01,10,100)
y2 = np.sin(x2)
x3 = np.random.randn(100)
y3 = np.linspace(0,3,100)
x4 = np.arange(0,6,0.5)
y4 = np.power(x4,3)
fig,ax = plt.subplots(2,2) #可以取'none','row','col','all'
ax1 = plt.subplot(221)
ax1.plot(x1,y1)
ax2= plt.subplot(222)
ax2.plot(x2,y2)
ax3 = plt.subplot(223)
ax3.scatter(x3,y3)
ax4 =plt.subplot(224,sharex=ax1) #改sharey试试效果
ax4.scatter(x4,y4)
plt.autoscale(enable=True,axis='both',tight=True) #调整坐标轴范围
plt.show()

#==============================================================================
# 7.3 ggplot用法
#==============================================================================
# 散点图
#读取数据
gp = pd.read_csv('hr_year.csv',index_col=0)
gg = ggplot(gp,aes(x='yearID',y='HR')) + geom_point(color='red',size=12)+ ggtitle('HR变化趋势') \
+ labs(x="年限",y="HR数量")
print(gg)

# 折线图
p = ggplot(aes(x='yearID',y='HR'),data=gp)+ geom_line(color='blue')+ggtitle('HR变化趋势') \
+ labs(x="年限",y="HR数量")
print (p)

# 分组散点图
GDP = pd.read_excel('Industry_GDP.xlsx')

p=ggplot(aes(x='Quarter',y='GDP',color='Industry_Type'),data=GDP) + geom_line(size=2) \
  + ggtitle('三个产业变化趋势')
print (p)

#p=ggplot(aes(x='Quarter',y='GDP',group='Industry_Type'),data=GDP) + geom_line(size=2)+ \
 # facet_grid('Industry_Type') + ggtitle('三个产业变化趋势') #facet_wrap('cyl')

# # #条形图
titanic_train = pd.read_csv('titanic_train.csv')
# 统计仓位等级数量
ggplot(aes(x='Pclass'),data=titanic_train) + geom_bar(fill ='red')  + labs(x="仓位等级")+\
ggtitle('不同仓位等级')

# 密度图
ggplot(aes(x='Age'),data=titanic_train) + geom_density(color ='red')  + labs(x="年龄")+\
ggtitle('年龄分布图')

# 分组密度图
ggplot(aes(x='Age',color='Sex'),data=titanic_train) + geom_density(size=2)  + labs(x="年龄")+\
ggtitle('不同性别年龄分布图')

# 直方图
ggplot(aes(x='Age'),data=titanic_train) + geom_histogram(fill ='red',color='black')  + labs(x="年龄")+\
ggtitle('年龄分布图')

# 箱线图
sec_buildings = pd.read_excel('sec_buildings.xlsx')
ggplot(aes(x ='region',y='price_unit'),data=sec_buildings) + geom_boxplot()  + labs(x="不同地区")+\
ggtitle('不同地区房价箱线图')

