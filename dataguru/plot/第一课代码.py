
"""这是第一章代码"""
# 1 python基本绘图
#==============================================================================
# 1.2 相关参数
#==============================================================================
# # # # # # matplotlib库中常见函数
# 绘制一个简单的图形，折线图是最基本的图形
# 导入相关库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir(r'D:\炼数成金数据可视化\数据可视化课件\第一课')
# 简单图形代码
x = np.linspace(0,10,100)
y = np.sin(x)
plt.plot(x,y)
plt.show()

# 添加标签和图例
x = np.arange(0,1.1,0.01)
y = x**2
plt.figure(figsize=(9,9),dpi=80) #确定画布大小，dpi:图形分辨率
plt.title('lines') #添加标题
plt.xlabel('x1')
plt.ylabel('y')
plt.xlim((0,1)) # 确定x轴的范围
plt.ylim((0,1)) # 确定x轴的范围
plt.xticks([0,0.2,0.4,0.6,0.8,1]) #确定x轴的刻度
plt.yticks([0,0.2,0.4,0.6,0.8,1]) #确定y轴的刻度
plt.plot(x,y, label='y =x^2')
plt.legend(loc='best') # 图例
plt.show()

# # # # # #相关参数
#  增加元素
x = np.linspace(0,10,100)
y = np.sin(x)
plt.plot(x,y,ls=':',lw=2,label='x和y的关系')
plt.legend()
plt.show()

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x = np.linspace(0,10,100)
y = np.sin(x)
plt.plot(x,y,ls=':',lw=2,label='x和y的关系')
plt.legend(loc= 'upper center')
plt.show()
# ls- -函数线条风格(='-' 实线, '--' 虚线 ,'-.' 点划线 ,':' 实点线)

# 调整线条样式，宽度，形状和点
# marker线条上点的形状
# markersize 点的大小
# c 颜色
# markeredgecolor  点的边框色
# markerfacecolor  点的填充色
x = np.linspace(0,10,100)
y = np.sin(x)
#plt.plot(x,y,ls=':',lw=2,marker='D',markersize=2,c= 'r',label='x和y的关系')
plt.plot(x,y,ls='--',lw=2,marker='s',markersize=10,c= 'red',markeredgecolor ='blue',markerfacecolor='black',label='x和y的关系')
plt.legend(loc= 'center')
plt.show()

#简单案例
#案例
wechat = pd.read_excel('wechat.xlsx')
wechat.Date=pd.to_datetime(wechat.Date,format='%Y-%m-%d')
# 绘制单条折线图
plt.plot(wechat.Date, # x轴数据
         wechat.Counts, # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 2, # 折线宽度
         color = 'steelblue', # 折线颜色
         marker = 'o', # 折线图中添加圆点
         markersize = 6, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='red') # 点的填充色
# 添加y轴标签
plt.ylabel('人数')
plt.xticks(rotation=45)
# 添加图形标题
plt.title('每天微信文章阅读人数趋势')
# 显示图形
plt.show()


# # # # # # 保存图形
x = np.arange(0,1.1,0.01)
y = x**2
plt.figure(figsize=(9,9),dpi=80) #确定画图大小，dpi:图形分辨率
plt.title('lines') #添加标题
plt.xlabel('x')
plt.ylabel('y')
plt.xlim((0,1)) # 确定x轴的范围
plt.ylim((0,1)) # 确定x轴的范围
plt.xticks([0,0.2,0.4,0.6,0.8,1]) #确定x轴的刻度
plt.yticks([0,0.2,0.4,0.6,0.8,1]) #确定y轴的刻度
plt.plot(x,y,label='y =x^2')
plt.legend(loc='best')
plt.savefig('D:\炼数成金数据可视化\数据可视化课件\第一课\\可视化.pdf')
plt.show()
