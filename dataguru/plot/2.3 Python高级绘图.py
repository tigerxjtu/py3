"""
这是第二章第3节的知识点
Python数据可视化
"""
# 2.3 python高级作图
# 导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# 支持中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
os.chdir('E:\python海量数据培训\python培训代码\Python数据可视化')
pd.set_option('display.max_columns',8)
#==============================================================================
# 2.3.1 图形样式
#==============================================================================
# # # # # # 绘制双坐标(双Y)
x = np.arange(0., np.e, 0.01)
y1 = np.exp(-x)
y2 = np.log(x)
fig = plt.figure() # 创建作图对象
ax1 = fig.add_subplot(111)
ax1.plot(x, y1)
ax1.set_ylabel('Y values for exp(-x)')
ax1.set_title("Double Y axis")
ax2 = ax1.twinx()  # this is the important function #添加次坐标轴
ax2.plot(x, y2, 'r')
ax2.set_xlim([0, np.e])
ax2.set_ylabel('Y values for ln(x)')
ax2.set_xlabel('Same X for both exp(-x) and ln(x)')
plt.show()

# # # 实际案例
GDP_data = pd.read_excel('国民经济核算季度数据.xlsx')
fig =plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(GDP_data['序号'],GDP_data['国内生产总值_当季值(亿元)'],label='GDP',c= 'r')
ax1.set_ylabel('国内生产总值_当季值(亿元)')
ax1.set_title("国内生产总值_当季值(亿元)和工业增加值_当季值(亿元)变化趋势")
plt.xticks(range(0,70,4),GDP_data.iloc[range(0,70,4),1],rotation=45)
ax1.legend(loc='upper left')
ax2 = ax1.twinx() #设置次坐标
ax2.plot(GDP_data['序号'],GDP_data['工业增加值_当季值(亿元)'],label='工业增加值',c='g')
ax2.set_ylabel('工业增加值_当季值(亿元)')
ax2.set_xlabel('时间')
ax2.legend(loc='upper center')
plt.show()

# # # # # # 添加注释
fig1 = plt.figure('1')
x = np.linspace(0,10,40)
y = np.random.randn(40)
plt.plot(x,y,ls='--',lw=2,marker='o',ms=20,mfc='orange',alpha=0.6)
plt.grid(ls=':',color='gray',alpha=0.5)
plt.text(6,0,'Matplotlib',size=30,rotation=30,bbox=dict(boxstyle='round',ec='#8968CD',fc='#FFE1FF'))
# ec--边缘颜色，fc--代表填充色，boxstyle--样式
fig1.show()

# 水印效果
fig2 = plt.figure('2')
x = np.linspace(0,10,40)
y = np.random.randn(40)
plt.plot(x,y,ls='--',lw=2,marker='o',ms=20,mfc='orange',alpha=0.6)
plt.grid(ls=':',color='gray',alpha=0.5) #alpha是透明度
plt.text(1,2,'Matplotlib',size=30,rotation=50,color='gray',alpha=0.5)
fig2.show()

# 以上是同时显示两幅图
## xy控制的是箭头坐标，xytext控制的是文本坐标,'data'表示和折线图使用的是相同的坐标系统
x = np.linspace(0.5,3.5,100)
y = np.sin(x)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.plot(x,y,c='b',ls='--',lw=2)
ax.annotate('maximum',xy=(np.pi/2,1),xycoords = 'data',
            xytext =((np.pi/2)+0.15,0.8),textcoords='data',
            weight='bold',color='r',
            arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='r'))
ax.text(2.8,0.4,'$y=\sin(x)$',fontsize=20,color='b',bbox=dict(facecolor='y',alpha=0.5))
plt.show()

# # # 有弧度的注解
x = np.linspace(0,10,2000)
y = np.sin(x)*np.cos(x)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y,c='c',ls='--',lw=2)
bbox=dict(boxstyle ='round',fc='#7EC0EE',edgecolor='#9B30EF') # 控制样式和颜色参数
arrowprops =dict(arrowstyle='-|>',
            connectionstyle='angle,angleA=0,angleB=90,rad=10',color='r') #箭头参数
ax.annotate('single point',(5,np.sin(5)*np.cos(5)),
            xytext =(3,np.sin(3)*np.cos(3)),fontsize=15,color='r',
            arrowprops=arrowprops, bbox=bbox)
ax.grid(ls=':',color='gray', alpha=0.6)
plt.show()

# # # 趋势线
x = np.linspace(0,10,2000)
y = np.sin(x)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y,c='c',ls='--',lw=2)
ax.set_ylim(-1.5,1.5)
arrowprops =dict(arrowstyle='-|>',color='r')
ax.annotate('',(3*np.pi/2,np.sin(3*np.pi/2)+0.05), #代表箭头位置
            xytext =(np.pi/2,np.sin(np.pi/2)+0.05),color='r',arrowprops=arrowprops) #添加趋势线，文本位置
ax.arrow(0,-0.4,np.pi/2,1.2,head_width=0.05,head_length=0.1,fc='g',ec='g')#添加趋势线
# 0和-0.4代表起始位置,后面两个分别代表x和y的水平增量
ax.grid(ls=':',color='gray',alpha=0.6)
plt.show()

# # # # # # 投影效果
x = np.linspace(0.5,3.5,100)
y = np.sin(x)
fig =plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
box = dict(facecolor='#6959CD',pad=2,alpha=0.4)
ax.plot(x,y,c='b',ls='--',lw=2)
title = '$y=\sin({x})$'
xaxis_label="$x\_axis$"
yaxis_label="$y\_axis$"
ax.set_xlabel(xaxis_label,fontsize=18,bbox=box)
ax.set_ylabel(yaxis_label,fontsize=18,bbox=box)
ax.set_title(title,fontsize=23,va='bottom') #控制位置的
ax.yaxis.set_label_coords(-0.05,0.8) # axes 坐标
ax.xaxis.set_label_coords(1.0,-0.05)# axes 坐标移动位置
ax.grid(ls=':',lw=1,color='gray',alpha=0.5)
plt.show()

# 实例
GDP_data = pd.read_excel('国民经济核算季度数据.xlsx')
xaxis_label="季度"
yaxis_label="GDP(产值)"
title = 'GDP变化趋势'
box = dict(facecolor='#6959CD',pad=2,alpha=0.4) # 控制坐标轴的参数
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(GDP_data['序号'],GDP_data['国内生产总值_当季值(亿元)'],c= 'c')
ax.set_xlabel(xaxis_label,fontsize=18,bbox=box)
ax.set_ylabel(yaxis_label,fontsize=18,bbox=box)
ax.set_title(title,fontsize=23,va='baseline')
ax.yaxis.set_label_coords(-0.05,0.5) # axes 坐标
ax.xaxis.set_label_coords(1.0,-0.05)# axes 坐标移动位置
plt.xticks(range(0,70,4),GDP_data.iloc[range(0,70,4),1],rotation=45)
ax.grid(ls=':',lw=1,color='gray',alpha=0.5)
plt.show()

#==============================================================================
# 2.3.2 划分画图主要函数应用
#==============================================================================
# # # # # # subplot使用方法
# 读取相关数据
GDP_data = pd.read_excel('国民经济核算季度数据.xlsx')
GDP = pd.read_excel('Province GDP 2017.xlsx')
Titanic = pd.read_csv('titanic_train.csv')
Titanic.dropna(subset=['Age'], inplace=True)
iris = pd.read_csv('iris.csv')
# 设置绘图区域
plt.subplot(121) # 位置
plt.plot(GDP_data['序号'],GDP_data['国内生产总值_当季值(亿元)'],c= 'c')
plt.xlabel("季度",fontsize=18)
plt.ylabel('GDP(产值)',fontsize=18)
plt.xticks(range(0,70,4),GDP_data.iloc[range(0,70,4),1],rotation=45)
plt.subplot(122)
plt.plot(GDP_data['序号'],GDP_data['工业增加值_当季值(亿元)'],c= 'c')
plt.xlabel("季度",fontsize=18)
plt.ylabel('工业增加值_当季值(亿元)',fontsize=18)
plt.xaxis.set_label_coords(1.05,-0.05)
plt.xticks(range(0,70,4),GDP_data.iloc[range(0,70,4),1],rotation=45)
plt.show()
pd.set_option('display.max_column',8)

# 第二种方式
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(121)
ax1.plot(GDP_data['序号'],GDP_data['国内生产总值_当季值(亿元)'],c= 'c')
ax1.set_xlabel("季度",fontsize=18)
ax1.set_ylabel('GDP(产值)',fontsize=18,labelpad =12)
plt.xticks(range(0,70,4),GDP_data.iloc[range(0,70,4),1],rotation=45)
ax1.xaxis.set_label_coords(1.05,-0.05)
# 设置第二幅图位置
ax2 = fig.add_subplot(122)
ax2.plot(GDP_data['序号'],GDP_data['工业增加值_当季值(亿元)'],c= 'orange')
ax2.set_xlabel("季度",fontsize=18)
ax2.set_ylabel('工业增加值_当季值(亿元)',fontsize=18,labelpad =12)
plt.xticks(range(0,70,4),GDP_data.iloc[range(0,70,4),1],rotation=45)
ax2.xaxis.set_label_coords(1.05,-0.05)
plt.show()

# 第三种方式
font_style = dict(fontsize=20,weight='black')
fig, ax = plt.subplots(1,2,sharey=True,figsize=(16,9))
# subplot(121)
ax1 = ax[0]
# ax1 = fig.add_subplot(121)
ax1.plot(GDP_data['序号'],GDP_data['国内生产总值_当季值(亿元)'],c= 'c',ls=":")
ax1.set_xlabel("季度",fontsize=18)
ax1.set_ylabel('GDP(产值)',fontsize=18,labelpad =12)
ax1.set_xticks(range(0,69,4))  # 位置
ax1.set_xticklabels(GDP_data.iloc[range(0,69,4),1],rotation=45)
ax1.xaxis.set_label_coords(1.05,-0.05)
# 设置第二幅图位置
ax2 = ax[1]
#ax2 = fig.add_subplot(122)
ax2.plot(GDP_data['序号'],GDP_data['工业增加值_当季值(亿元)'],c= 'orange')
ax2.set_xlabel("季度",fontsize=18)
ax2.set_ylabel('工业增加值_当季值(亿元)',fontsize=18,labelpad =12)
plt.xticks(range(0,70,4),GDP_data.iloc[range(0,70,4),1],rotation=45)
ax2.xaxis.set_label_coords(1.05,-0.05)
plt.suptitle('GDP和工业增加值变化趋势',**font_style)
plt.show()

# # # 在非等分画布的绘图区域上实现图形展示
# 读取Titanic数据
Titanic = pd.read_csv('titanic_train.csv')
Titanic.dropna(subset=['Age'], inplace=True)
#作图
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(GDP_data['序号'],GDP_data['国内生产总值_当季值(亿元)'],c= 'c')
ax1.set_xlabel("季度",fontsize=18)
ax1.set_ylabel('GDP(产值)',fontsize=18,labelpad =12)
plt.xticks(range(0,70,4),GDP_data.iloc[range(0,70,4),1],rotation=45)
ax1.xaxis.set_label_coords(1.05,-0.05) # 调整标签位置
ax1.set_title('国内生产总值趋势')
# 设置位置
ax2 = fig.add_subplot(222)
ax2.hist(x = Titanic.Age, bins=20,color='c',edgecolor ='black',density=True)
ax2.set_xlabel('年龄',fontsize =15)
ax2.set_ylabel('频数',fontsize =15)
ax2.set_title('乘客年龄分布图')
# 设置位置
colors= ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#00FFFF']
ax3 = fig.add_subplot(224)
ax3.bar(GDP.index.values,GDP.GDP,width=0.6,align='center',color= colors,
        tick_label=GDP.Province)
ax3.set_xlabel('省份',fontsize = 20,labelpad =15)
ax3.set_ylabel('GDP产值(万亿)',fontsize = 20,labelpad =15)
# 添加表格
col_labels = ['GDP(万亿)']
row_labels = GDP.Province
table_vals =np.array(GDP.GDP.values).reshape(-1,1)
col_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#00FFFF']
my_table = plt.table(cellText=table_vals,cellLoc='center' ,colWidths=[0.1] * 6,
   rowLabels=row_labels, colLabels=col_labels,rowColours=col_colors,bbox=[0.8,0.7,0.1,0.25])
# 显示
plt.show()

# # # # # # subplot2grid应用
# 案例
# 本案例中将绘图区域划分为6个子区
labels =["A难度水平",'B难度水平','C难度水平','D难度水平']
students = [0.35,0.15,0.20,0.30]
colors = ['red','green','blue','yellow']
explode = (0.1,0.1,0,0)
plt.subplot2grid((1,3),(0,0)) #设置绘图区域
plt.pie(students,explode = explode,labels =labels,autopct='%1.1f%%',startangle=45,shadow=True,
        colors=colors)
# 设置x，y轴刻度一致，这样饼图才能是圆的
plt.axis('equal')
plt.title('选择不同难度测试试卷的学生百分比')
# 设置位置
plt.subplot2grid((1,3),(0,1)) #设置绘图区域
plt.hist(x = Titanic.Age, bins=20,color='c',edgecolor ='black',density=True)
plt.xlabel('年龄',fontsize =15)
plt.ylabel('频数',fontsize =15)
plt.title('乘客年龄分布图')
# 设置位置
colors= ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#00FFFF']
plt.subplot2grid((1,3),(0,2)) #设置绘图区域
plt.bar(GDP.index.values,GDP.GDP,width=0.6,align='center',color= colors,
        tick_label=GDP.Province)
plt.xlabel('省份',fontsize = 20,labelpad =15)
plt.ylabel('GDP产值(万亿)',fontsize = 20,labelpad =15)
# 添加表格
col_labels = ['GDP(万亿)']
row_labels = GDP.Province
table_vals =np.array(GDP.GDP.values).reshape(-1,1)
col_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#00FFFF']
my_table = plt.table(cellText=table_vals,cellLoc='center' ,colWidths=[0.1] * 6,
   rowLabels=row_labels, colLabels=col_labels,rowColours=col_colors,bbox=[0.8,0.7,0.1,0.25])
# 显示
plt.suptitle('不同图形的展示',size=20)
plt.show()

# 使用gridspec类
from matplotlib.gridspec import GridSpec
labels =["A难度水平",'B难度水平','C难度水平','D难度水平']
students = [0.35,0.15,0.20,0.30]
colors = ['red','green','blue','yellow']
explode = (0.1,0.1,0,0)
box = {'facecolor':'lightgreen','pad':3,'alpha':0.2}
fig = plt.figure(figsize=(16,9))
gs = GridSpec(2,2) #相当于两行两列
ax1 = fig.add_subplot(gs[0,:]) #绘制在第一行
ax1.pie(students,explode = explode,labels =labels,autopct='%1.1f%%',startangle=45,shadow=True,
        colors=colors)
ax1.axis('equal')
ax1.set_title('选择不同难度测试试卷的学生百分比')
# 第二个子图位置
ax2 = fig.add_subplot(gs[1,0]) #绘制在第2行和第一列
ax2.scatter(x = iris.Petal_Width,y = iris.Petal_Length,s =20,marker='s',lw=2,
            color ='grey',edgecolors='k')
ax2.set_xlabel('花瓣宽度',bbox =box)
ax2.set_ylabel('花瓣长度',bbox =box)
ax2.set_title('鸢尾花花瓣宽度和长度关系图')
for ticklabel in ax2.get_xticklabels():
    ticklabel.set_rotation(45)
ax2.yaxis.set_label_coords(-0.04,0.5)
ax2.xaxis.set_label_coords(0.5,-0.12)
# 第三个子图位置
colors= ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#00FFFF']
ax3 = fig.add_subplot(gs[1,1]) #绘制在第2行和第2列
ax3.bar(GDP.index.values,GDP.GDP,width=0.6,align='center',color= colors,
        tick_label=GDP.Province)
ax3.set_xlabel('省份',fontsize = 20,labelpad =15)
ax3.set_ylabel('GDP产值(万亿)',fontsize = 20,labelpad =15)
# 添加表格
col_labels = ['GDP(万亿)']
row_labels = GDP.Province
table_vals =np.array(GDP.GDP.values).reshape(-1,1)
col_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#00FFFF']
my_table = plt.table(cellText=table_vals,cellLoc='center' ,colWidths=[0.1] * 6,
   rowLabels=row_labels, colLabels=col_labels,rowColours=col_colors,bbox=[0.8,0.7,0.1,0.25])
# 显示
gs.tight_layout(fig) #控制子图参数的
plt.show()
#==============================================================================
# 2.3.3 共享绘图区域的坐标轴
#==============================================================================
# # # # # # 共享单一区域坐标轴位置

jd_stock  = pd.read_csv('data.csv', sep =',',header=None,names =['name','date','opening_price','closing_price',
                                                                 'lowest_price','highest_price','volume'])
# jd_stock['date'] = pd.to_datetime(jd_stock['date'])
fig, ax1 =plt.subplots() #创建画图对象
ax1.plot(jd_stock['date'],jd_stock['opening_price'],c='b',ls='--',label='开盘价')
ax1.legend(loc='upper right')
ax1.set_xticks(range(0,70,4))  # 位置
ax1.set_xticklabels(jd_stock.iloc[range(0,70,4),1],rotation=45)
ax1.xaxis.set_label_coords(1.05,-0.02)
ax1.set_xlabel('日期',fontsize=10)
ax1.set_ylabel('开盘价',color='b')
ax1.tick_params('y',colors='b') #使坐标轴的和线条相匹配
ax2 =ax1.twinx() #使用子坐标
ax2.plot(jd_stock['date'],jd_stock['volume'],c='r',ls=':',label='成交量')
ax2.set_xticks(range(0,70,4))  # 位置
ax2.set_xticklabels(jd_stock.iloc[range(0,70,4),1],rotation=45)
ax2.legend(loc='upper left')
ax2.set_xlabel('日期',fontsize=10)
ax2.set_ylabel('成交量',color='r')
ax2.tick_params('y',colors='r') #使坐标轴的和线条相匹配
plt.show()

# # # # # # 共享不同区域的坐标轴
x1 = np.linspace(0,2*np.pi,400)
y1 = np.cos(x1**2)
x2 = np.linspace(0.01,10,100)
y2 = np.sin(x2)
x3 = np.random.randn(100)
y3 = np.linspace(0,3,100)
x4 = np.arange(0,6,0.5)
y4 = np.power(x4,3)
# fig,ax = plt.subplots(2,2)
fig,ax = plt.subplots(2,2,sharex = 'all') #可以取'none','row','col','all'
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
fig.subplots_adjust(hspace = 0.2) #去掉空间
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
