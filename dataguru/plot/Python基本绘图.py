"""
这是第二章第1节的知识点
Python数据可视化
"""
# 2.1  python基本绘图
#==============================================================================
# 2.1.1 matplotlib绘图基本语法
#==============================================================================
# # # # # # matplotlib库中常见函数
# 绘制一个简单的图形，折线图是最基本的图形
# 导入相关库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir(r'E:\python海量数据培训\python培训代码\Python数据可视化')

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

#案例
wechat = pd.read_excel('wechat.xlsx')
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
plt.savefig('E:\python海量数据培训\python培训代码\Python数据可视化\\可视化.pdf')
plt.show()

#==============================================================================
# 2.1.2 简单图形绘制
#==============================================================================
# # # # # # 饼图
kinds = ['简易箱','保温箱','行李箱','密封箱']
soldNums = [0.05,0.45,0.15,0.35]
explodes = [0,0.1,0.1,0]
plt.axes(aspect='equal')
plt.pie(x=soldNums, labels=kinds, explode=explodes, \
        autopct ='%.2f%%', wedgeprops= { 'linewidth' : 1.0 , 'edgecolor' : 'green'}, startangle=0) #设置百分比格式
plt.title('不同类型箱子的销售数量占比')
plt.show()

# 添加修饰的饼图
# 构造数据
edu = [0.2515,0.3724,0.3336,0.0368,0.0057]
labels = ['中专','大专','本科','硕士','其他']
explode = [0,0.1,0,0,0]  # 生成数据，用于突出显示大专学历人群
colors=['#9999ff','#ff9999','#7777aa','#2442aa','#dd5555']  # 自定义颜色
# 中文乱码和坐标轴负号的处理
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 将横、纵坐标轴标准化处理，确保饼图是一个正圆，否则为椭圆
plt.axes(aspect='equal')
# 绘制饼图
plt.pie(x = edu, # 绘图数据
        explode=explode, # 突出显示大专人群
        labels=labels, # 添加教育水平标签
        colors=colors, # 设置饼图的自定义填充色
        autopct='%.1f%%', # 设置百分比的格式，这里保留一位小数
        pctdistance=0.5,  # 设置百分比标签与圆心的距离
        labeldistance = 1.1, # 设置教育水平标签与圆心的距离
        startangle = 120, # 设置饼图的初始角度
        radius = 1.2, # 设置饼图的半径
        counterclock = False, # 是否逆时针，这里设置为顺时针方向
        wedgeprops = {'linewidth': 1.5, 'edgecolor':'green'},# 设置饼图内外边界的属性值
        textprops = {'fontsize':10, 'color':'red'}, # 设置文本标签的属性值
        )
# 添加图标题
plt.title('失信用户的受教育水平分布',pad =30)
# 显示图形
plt.show()

# # # 使用pandas
# 导入第三方模块
import pandas as pd
# 构建序列
data1 = pd.Series({'中专':0.2515,'大专':0.3724,'本科':0.3336,'硕士':0.0368,'其他':0.0057})
# 将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data1.name = ''
# 控制饼图为正圆
plt.axes(aspect = 'equal')
# plot方法对序列进行绘图
data1.plot(kind = 'pie', # 选择图形类型
           autopct='%.1f%%', # 饼图中添加数值标签
           radius = 1, # 设置饼图的半径
           startangle = 180, # 设置饼图的初始角度
           counterclock = False, # 将饼图的顺序设置为顺时针方向
           title = '失信用户的受教育水平分布', # 为饼图添加标题
           wedgeprops = {'linewidth': 1.5, 'edgecolor':'green'}, # 设置饼图内外边界的属性值
           textprops = {'fontsize':16, 'color':'black'} # 设置文本标签的属性值
          )
# 显示图形
plt.show()

# # # # # # 条形图
# 绘制垂直条形图
# 创建简单数据
x =[1,2,3,4,5,6,7,8]
y =[3,1,4,5,8,9,7,2]
labels = ['q','a','c','e','r','j','b','p']
plt.bar(x,y,align='center',color='y',tick_label=labels,hatch='--')
plt.xlabel('箱子编号',labelpad =19)  # 控制标签和坐标轴的距离
plt.ylabel('箱子重量(kg)',labelpad =10)
plt.title('不同箱子的重量图',pad=15)
plt.show()

# 绘制水平条形图
# 创建简单数据
x =[1,2,3,4,5,6,7,8]
y =[3,1,4,5,8,9,7,2]
labels = ['q','a','c','e','r','j','b','p']
plt.barh(x,y,align='center',color='r',tick_label=labels)
plt.xlabel('箱子重量(kg)',fontsize=20,labelpad =19)
plt.ylabel('箱子编号',fontsize=20,labelpad =19)
plt.title('不同箱子的重量图',fontsize=20,pad =19)
plt.show()

#使用pandas绘图
x = ['q','a','c','e','r','j','b','p']
y =[3,1,4,5,8,9,7,2]
df = pd.DataFrame({'x':x,'y':y},index = range(1,9))
df.y.plot(kind = 'bar', width = 0.8, rot = 0, color = 'steelblue', title = '不同箱子的重量图')
# 添加y轴标签
plt.ylabel('箱子重量(kg)')
# 添加x轴刻度标签
plt.xticks(range(len(df.x)), #指定刻度标签的位置
          df.x) # 指出具体的刻度标签值
# 为每个条形图添加数值标签
plt.text(1,1.2,'%s' % round(1,1),va='center')
for i,j in enumerate(df.y):
    plt.text(i,j+0.2,'%s' % round(j,1) ,va='center')
# 显示图形
plt.show()

# # # # # # 直方图
#读取数据
import  os
os.chdir('E:\python海量数据培训\python培训代码\Python数据可视化')
#导入数据
Titanic = pd.read_csv('titanic_train.csv')
#检查年龄是否有缺失
any(Titanic['Age'].isnull())
# 删除缺失值
Titanic['Age'].dropna(inplace=True)
#绘制直方图
plt.hist(x =Titanic.Age,bins=30,color='r',edgecolor='black', rwidth=2,normed=True)
# plt.xlabel('年龄',fontsize =15,labelpad =20)
# plt.ylabel('频数',fontsize =15,labelpad =20)
plt.title('年龄分布图',fontsize =15,pad =20)
plt.show()

# 使用pandas绘图
Titanic.Age.plot(kind = 'hist', bins = 20, color = 'steelblue', edgecolor = 'black', label = '直方图',density = True)
# 绘制核密度图
Titanic.Age.plot(kind = 'density', color = 'red',label = '核密度图',xlim=[0,Titanic.Age.max()+5])
# 添加x轴和y轴标签
plt.xlabel('年龄')
plt.ylabel('核密度值')
# 添加标题
plt.title('乘客年龄分布')
# 显示图例
plt.legend()
# 显示图形
plt.show()

# # # # # # 散点图
# 读取数据
creditcard_exp  = pd.read_csv('creditcard_exp.csv',skipinitialspace = True)
creditcard_exp.dropna(subset=['Income','avg_exp'],inplace=True)
# 画图
plt.scatter(x = creditcard_exp.Income,y=creditcard_exp.avg_exp,color= 'steelblue',marker='o', s=100)
plt.xlabel('收入',fontsize=12) # 坐标轴标签大小
plt.ylabel('支出',fontsize=12)
plt.title('收入与支出关系')
plt.show()

#pandas模块
creditcard_exp.plot(x= 'Income',y='avg_exp',kind='scatter',marker='o',s=100,title='收入与支出散点图')
plt.xlabel('收入',fontsize=12) #坐标轴标签大小
plt.ylabel('支出',fontsize=12)
plt.title('收入与支出关系')
plt.show()

#==============================================================================
# 2.1.3 图形设置
#==============================================================================
# # # # # # 图例用法
pd.set_option('display.max_columns',8)
jd_stock  = pd.read_csv('data.csv', sep =',',header=None,names =['name','date','opening_price','closing_price',
                                                                 'lowest_price','highest_price','volume'])
jd_stock['date'] = pd.to_datetime(jd_stock['date'])
plt.plot(jd_stock['opening_price'],label='Opening_Price')
plt.plot(jd_stock['closing_price'],label='Closing_Price')
plt.legend(loc='lower right', frameon=False) #控制是否有边框
plt.show()

#调整位置
plt.plot(jd_stock['opening_price'],label='Opening_Price')
plt.plot(jd_stock['closing_price'],label='Closing_Price')
plt.legend(loc='best', frameon=False,fontsize = 12) #控制是否有边框和调整字体
plt.show()

# # # # # # 图像大小
plt.plot(jd_stock['opening_price'],label='Opening_Price')
plt.plot(jd_stock['closing_price'],label='Closing_Price')
plt.legend(loc='upper right',frameon = True)
plt.xticks([xtick for xtick in range(71)],
           list(jd_stock['date']),rotation =40)
plt.xlabel(fontsize=5)
fig = plt.gcf() #返回当前图像并设置为该图像对象名称为fig
fig.set_size_inches(18.5,10.5)
plt.show()

# # # # # # 绘制网格线
# plt.grid(ls,c), ls=线条风格,c线条颜色
plt.plot(jd_stock['opening_price'],label='Opening_Price')
plt.plot(jd_stock['closing_price'],label='Closing_Price')
plt.legend(loc='upper right',frameon = True)
plt.grid(ls='--',c = 'darkblue')
plt.show()

# # # # # # 绘制平行于x轴和y轴的水平参考线
# plt.axhline(y,c,ls,lw)
x = np.linspace(0.05,10,1000)
y = np.sin(x)
plt.plot(x,y,ls='-.',lw=2,c='c',label='plot figure')
plt.legend()
plt.axhline(y = 0,c = 'r',ls = '--',lw = 2)
plt.axvline(x = 4,c = 'r',ls = '--',lw = 2)
plt.show()

# # # # # # 绘制平行于x轴和y轴的参考区域
# plt.axvspan(xmin,xmax,faceclor,alpha)
x = np.linspace(0.05,10,1000)
y = np.sin(x)
plt.plot(x,y,ls='-.',lw=2,c='c',label='plot figure')
plt.legend()
plt.axvspan(xmin=4,xmax=6, facecolor='y',alpha=0.6)
plt.axhspan(ymin=0,ymax=0.5,facecolor='y',alpha=0.3)
plt.show()
# # # # # # 添加图形内容,无指向
# plt.text(x,y,string,weight,color)
# weight --文本风格粗细风格
plt.plot(jd_stock['opening_price'],label='Opening_Price')
plt.plot(jd_stock['closing_price'],label='Closing_Price')
plt.legend(loc='lower center',frameon = True)
plt.text(20,32,'开盘价和收盘价走势图',weight ='bold',color='red',fontsize=12)
plt.show()

# # # # # # 添加图形内容细节的指向型注解
x = np.linspace(0.05,10,1000)
y = np.sin(x)
plt.plot(x,y,ls='-.',lw=2,c='c',label='plot figure')
plt.legend()
plt.annotate('maximum',xy=(np.pi/2,1), xytext=((np.pi/2)+1,0.8),weight='bold',color='b',
             arrowprops =dict(arrowstyle='->',connectionstyle = 'arc3',color='r'))
plt.show()

# # # #用于之前的数据
plt.plot(jd_stock['opening_price'],label='Opening_Price')
plt.plot(jd_stock['closing_price'],label='Closing_Price')
plt.legend(loc='lower center',frameon = True,fontsize =15)
plt.annotate('开盘价',xy=(16,28.80), xytext=(20,30),weight='bold',color='b',
             arrowprops =dict(arrowstyle='->',connectionstyle = 'arc3',color='b'))
plt.annotate('收盘价',xy=(53,30.32), xytext=(59,28),weight='bold',color='r', fontsize = 15,
             arrowprops = dict(arrowstyle='->',connectionstyle = 'arc3',color='r'))
plt.text(10,32,'开盘价和收盘价走势图',weight ='bold',color='red',fontsize=15)
plt.title('开盘价和收盘价股票走势图',color ='steelblue')
plt.show()

# # # # # # 创建子图
rad = np.arange(0,np.pi*2,0.01)
# 第一幅子图
p1 = plt.figure(figsize=(8,6),dpi=80)
ax1 = p1.add_subplot(2,1,1)
plt.title('lines')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim((0,1))
plt.ylim((0,1))
plt.xticks([0,0.2,0.4,0.6,0.8,1]) #确定x轴的刻度
plt.yticks([0,0.2,0.4,0.6,0.8,1]) #确定y轴的刻度
plt.plot(rad,2*rad,label ='y=2*x')
plt.legend(loc='upper right')

# 第二幅子图
ax2 = p1.add_subplot(2,1,2)
plt.title('sin/cos')
plt.xlabel('rad')
plt.ylabel('value')
plt.xlim((0,np.pi*2))
plt.ylim((-1,1))
plt.xticks([0,np.pi/2,np.pi,np.pi*1.5,np.pi*2])
plt.yticks([-1,-0.5,0,0.5,1])
plt.plot(rad, np.cos(rad), label='sin(x)')
plt.legend(loc='upper center')
plt.show()



