"""
这是第二章第2节的知识点
Python数据可视化
"""
# 2.2 python统计图形
# 导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#==============================================================================
# 2.2.1常见图形绘制
#==============================================================================
# # # # # #柱状图
# 简单柱状图
# 读取数据
# fig代表绘图窗口（Figure），ax代表这个绘图窗口上的坐标系（axes）。后面的ax.xxx则是表示对ax坐标系进行xxx操作
# fig, ax = plt.subplots(figsize=(15,10)) 创建一个绘图对象
os.chdir('E:\python海量数据培训\python培训代码\Python数据可视化')
GDP = pd.read_excel('Province GDP 2017.xlsx')
fig, ax = plt.subplots(figsize=(15,10)) #创建一个绘图对象
ax.bar(GDP.index.values,GDP.GDP,0.5) #0.5代表柱状的宽度
ax.set_xticks(GDP.index.values) #位置
ax.set_xticklabels(GDP.Province,rotation = 45) #给每个位置一个具体的标签
plt.xticks(fontsize=15 )
plt.yticks(fontsize=15)
ax.set_xlabel('省份',fontsize = 20)
ax.set_ylabel('GDP产值(万亿)',fontsize = 20)
ax.set_title('2017年6个省份的GDP',fontsize = 25)
plt.show()
# # # 堆叠图
industry_GDP = pd.read_excel('Industry_GDP.xlsx')
temp = pd.crosstab(industry_GDP['Quarter'],industry_GDP['Industry_Type'],values=industry_GDP['GDP'],aggfunc=np.sum)
plt.bar(x= temp.index.values,height= temp['第一产业'],color='steelblue',label='第一产业',tick_label = temp.index.values)
plt.bar(x= temp.index.values,height= temp['第二产业'],bottom =temp['第一产业'], color='green',label='第二产业',
        tick_label = temp.index.values)
plt.bar(x= temp.index.values,height = temp['第三产业'],bottom =temp['第一产业'] +  temp['第二产业'],color='red',label='第三产业',
        tick_label = temp.index.values)
plt.ylabel('生产总值(亿)')
plt.title('2017各季度三产业总值')
plt.legend(loc=2, bbox_to_anchor=(1.01,0.8)) #图例显示在外面
plt.show()

# # # 堆叠图占比
temp = pd.crosstab(industry_GDP['Quarter'],industry_GDP['Industry_Type'],values=industry_GDP['GDP'],aggfunc=np.sum)
temp = temp.div(temp.sum(1).astype(float), axis=0)
plt.bar(x= temp.index.values,height= temp['第一产业'],color='steelblue',label='第一产业',tick_label = temp.index.values)
plt.bar(x= temp.index.values,height= temp['第二产业'],bottom =temp['第一产业'], color='green',label='第二产业',
        tick_label = temp.index.values)
plt.bar(x= temp.index.values,height = temp['第三产业'],bottom =temp['第一产业'] +  temp['第二产业'],color='red',label='第三产业',
        tick_label = temp.index.values)
plt.ylabel('各产业占比')
plt.title('2017各季度三产业总值占比')
plt.legend(loc = 2,bbox_to_anchor=(1.01,0.8))
plt.show()

# # # 水平交错条形图
temp = pd.crosstab(industry_GDP['Quarter'],industry_GDP['Industry_Type'],values=industry_GDP['GDP'],aggfunc=np.sum)
bar_width = 0.2 #设置宽度
quarter = temp.index.values #取出季度名称
plt.bar(x= np.arange(0,4),height= temp['第一产业'],color='steelblue',label='第一产业',width = bar_width)
plt.bar(x= np.arange(0,4) + bar_width,height= temp['第二产业'], color='green',label='第二产业',width=bar_width)
plt.bar(x= np.arange(0,4) + 2*bar_width,height= temp['第三产业'], color='red',label='第三产业',width=bar_width)
plt.xticks(np.arange(4)+0.2,quarter,fontsize=12)
plt.ylabel('生产总值(亿)',fontsize=15)
plt.title('2017各季度三产业总值',fontsize=20)
plt.legend(loc = 'upper left')
plt.show()

# # # # # # 条形图
# # #简单条形图
# 对GDP数据进行升序排序
GDP = GDP.sort_values(by ='GDP')
plt.barh(y=range(GDP.shape[0]) ,width=GDP.GDP.values,color='darkblue',align='center',tick_label=GDP.Province.values)
plt.xlabel('GDP(万亿)',fontsize=12)
plt.ylabel('省份',fontsize=12)
plt.title('2017各季度三产业总值',fontsize=20)
plt.show()

# # # 条形堆叠图
industry_GDP = pd.read_excel('Industry_GDP.xlsx')
temp = pd.crosstab(industry_GDP['Quarter'],industry_GDP['Industry_Type'],values=industry_GDP['GDP'],aggfunc=np.sum)
plt.barh(y= temp.index.values,width= temp['第一产业'],color='steelblue',label='第一产业',tick_label = temp.index.values)
plt.barh(y= temp.index.values,width= temp['第二产业'],left =temp['第一产业'], color='green',label='第二产业',
        tick_label = temp.index.values)
plt.barh(y= temp.index.values,width = temp['第三产业'],left =temp['第一产业'] +  temp['第二产业'],color='red',label='第三产业',
        tick_label = temp.index.values)
plt.ylabel('生产总值(亿)')
plt.title('2017各季度三产业总值')
plt.legend(loc = 'lower right')
plt.show()

# # # 条形交错图
temp = pd.crosstab(industry_GDP['Quarter'],industry_GDP['Industry_Type'],values=industry_GDP['GDP'],aggfunc=np.sum)
bar_width = 0.2 #设置宽度
fig1 = plt.figure('fig1')
quarter = temp.index.values #取出季度名称
plt.barh(y= np.arange(0,4) ,width= temp['第一产业'],color='steelblue',label='第一产业',height =bar_width)
plt.barh(y= np.arange(0,4) + bar_width,width= temp['第二产业'], color='green',label='第二产业',height = bar_width )
plt.barh(y= np.arange(0,4) + 2*bar_width,width= temp['第三产业'], color='red',label='第三产业',height = bar_width )
plt.yticks(np.arange(4)+0.2,quarter,fontsize=12)
plt.xlabel('生产总值(亿)',fontsize=15,labelpad =10)
plt.title('2017各季度三产业总值',fontsize=20)
# plt.legend(loc = 'lower right')
plt.legend(loc=2, bbox_to_anchor=(1.01,0.7)) #图例显示在外面
# plt.show()
fig1.show()
# # # 条形图堆叠占比
fig2 = plt.figure('fig2')
temp = pd.crosstab(industry_GDP['Quarter'],industry_GDP['Industry_Type'],values=industry_GDP['GDP'],aggfunc=np.sum)
temp = temp.div(temp.sum(1).astype(float), axis=0)
plt.barh(y= temp.index.values,width = temp['第一产业'], color='steelblue',label='第一产业',tick_label = temp.index.values)
plt.barh(y= temp.index.values,width = temp['第二产业'], left=temp['第一产业'], color='green',label='第二产业',
        tick_label = temp.index.values,hatch='/////')
plt.barh(y= temp.index.values,width = temp['第三产业'],left =temp['第一产业'] +  temp['第二产业'],color='red',label='第三产业',
        tick_label = temp.index.values)
plt.ylabel('各产业占比')
plt.title('2017各季度三产业总值占比')
plt.legend(loc=2, bbox_to_anchor=(1.01,0.7)) #图例显示在外面
# plt.show()
fig2.show()

# # # # # # 直方图
# 读取Titanic数据
Titanic = pd.read_csv('titanic_train.csv')
Titanic.dropna(subset=['Age'], inplace=True)
# 绘制直方图
plt.hist(x = Titanic.Age, bins=20,color='c',edgecolor ='black',density=True)
plt.xlabel('年龄',fontsize =15)
plt.ylabel('频数',fontsize =15)
plt.title('乘客年龄分布图')
plt.show()

# 添加核密度图和正态分布图
# 定义正态分布概率密度公式
#normfun正态分布函数，mu: 均值，sigma:标准差，pdf:概率密度函数，np.exp():概率密度函数公式
def normfun(x,mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf
mean_x = Titanic.Age.mean()
std_x = Titanic.Age.std()
# x的范围为60-150，以1为单位,需x根据范围调试
x = np.arange(min(Titanic.Age), max(Titanic.Age)+10,1)
# x数对应的概率密度
y = normfun(x, mean_x, std_x)
plt.hist(x=Titanic.Age, bins=20,color='c',edgecolor ='black',label ='分布图',density=True)
plt.plot(x,y, color='g',linewidth = 3,label ='正态分布图') #正态分布图
Titanic['Age'].plot(kind='kde',color='red',xlim=[0,90],label='核密度图')
plt.xlabel('年龄',fontsize =15,labelpad=15)
plt.ylabel('频数',fontsize =15,labelpad=15)
plt.title('乘客年龄分布图')
plt.legend()
plt.show()

# # # # # #饼图
labels =["A难度水平",'B难度水平','C难度水平','D难度水平']
students = [0.35,0.15,0.20,0.30]
colors = ['red','green','blue','yellow']
explode = (0.1,0.1,0,0)
plt.pie(students,explode = explode,labels =labels,autopct='%3.2f%%',startangle=45,shadow=True,
        colors=colors)
# 设置x，y轴刻度一致，这样饼图才能是圆的
plt.axis('equal')
plt.title('选择不同难度测试试卷的学生百分比')
plt.show()

# 带图例的饼图
elements = ['面粉','砂糖','奶油','草莓酱','坚果']
weights =[40,15,20,10,15]
colors = ['#1b9e77', '#d95f02','#7570b3','#66a613','#e6ab02']
wedges,texts,autotexts = plt.pie(weights,autopct='%3.1f%%',textprops=dict(color='w'),colors=colors)
plt.legend(wedges,elements,fontsize=12,title='配比表',loc ='center',
           bbox_to_anchor=(0.7,0.2,0.2,0.2)) # 上下左右的
plt.setp(autotexts,size=15,weight='bold')
plt.setp(texts,size=15)
plt.axis('equal')
plt.title('果酱面包配料比例表',fontsize = 20)
plt.show()

# 绘制内嵌环形饼图
elements = ['面粉','砂糖','奶油','草莓酱','坚果']
weights1 =[40,15,20,10,15]
weights2 =[30,25,15,20,10]
outer_colors = ['#1b9e77', '#d95f02','#7570b3','#66a613','#e6ab02']
inner_colors = ['#1b9e77', '#d95f02','#7570b3','#66a613','#e6ab02']
wedges1,texts1,autotexts1 = plt.pie(weights1,autopct='%3.1f%%', radius =1,   pctdistance=0.85,
                                    colors=outer_colors,textprops=dict(color='w'),wedgeprops=dict(width=0.3,edgecolor='w'))
wedges2,texts2,autotexts2 = plt.pie(weights2,autopct='%3.1f%%', radius =0.7, pctdistance=0.75,
                                    colors=inner_colors,textprops=dict(color='w'),wedgeprops=dict(width=0.3,edgecolor='w'))
plt.legend(wedges1,elements,fontsize=12,title='配比表',loc ='center left',
           bbox_to_anchor=(0.9, 0.2, 0.2, 1))
plt.setp(autotexts1,size=15,weight='bold')
plt.setp(autotexts2,size=15,weight='bold')
plt.setp(texts1,size=12)
plt.title('不同果酱面包配料比例表',fontsize = 20)
plt.show()

# # # # # #箱线图
sec_building = pd.read_excel('sec_buildings.xlsx')
plt.boxplot(x=sec_building.price_unit,patch_artist=True,showmeans =True,
            boxprops={'color':'black','facecolor':'steelblue'},
            showfliers=True,
            flierprops={'marker':'o','markerfacecolor':'red','markersize':5},
            meanprops={'marker':'D','markerfacecolor':'indianred','markersize':4},
            medianprops={'linestyle':'--','color':'orange'},labels=[''])
plt.title('二手房价分布箱线图')
plt.show()

# 二手房在各行政区域的平均单价
group_region = sec_building.groupby('region')
avg_price = group_region.aggregate({'price_unit':np.mean}).sort_values('price_unit', ascending = False)

# 通过循环，将不同行政区域的二手房存储到列表中
region_price = []
for region in avg_price.index:
    region_price.append(sec_building.price_unit[sec_building.region == region])
# 绘制分组箱线图
plt.boxplot(x = region_price,
            patch_artist=True,
            labels = avg_price.index, # 添加x轴的刻度标签
            showmeans=True,
            boxprops = {'color':'black', 'facecolor':'steelblue'},
            flierprops = {'marker':'o','markerfacecolor':'red', 'markersize':3},
            meanprops = {'marker':'D','markerfacecolor':'indianred', 'markersize':4},
            medianprops = {'linestyle':'--','color':'orange'}
           )
# 添加y轴标签
plt.ylabel('单价（元）')
# 添加标题
plt.title('不同行政区域的二手房单价对比')
# 显示图形
plt.show()

# # # # # # 散点图
iris = pd.read_csv('iris.csv')
#绘制散点图
plt.scatter(x = iris.Petal_Width,y = iris.Petal_Length,s =10,
            color ='steelblue')
plt.xlabel('花瓣宽度')
plt.ylabel('花瓣长度')
plt.title('鸢尾花花瓣宽度和长度关系图')
plt.show()
# # # 绘制不同种类的散点图关系
plt.scatter(x = iris.Petal_Width[iris['Species'] =='setosa'],y = iris.Petal_Length[iris['Species'] =='setosa'],s =20,
            color ='steelblue',marker='o',label = 'setosa',alpha=0.8)
plt.scatter(x = iris.Petal_Width[iris['Species'] =='versicolor'],y = iris.Petal_Length[iris['Species'] =='versicolor'],s =30,
            color ='indianred',marker='s',label = 'versicolor')
plt.scatter(x = iris.Petal_Width[iris['Species'] =='virginica'],y = iris.Petal_Length[iris['Species'] =='virginica'],s =40,
            color ='green',marker='x',label = 'virginica')
plt.xlabel('花瓣宽度',fontsize=12)
plt.ylabel('花瓣长度')
plt.title('不同种类的鸢尾花花瓣宽度和长度关系图')
plt.legend(loc='upper left')
plt.show()


# 使用循环方式
colors_iris = ['steelblue','indianred','green']
sepcies =[ 'setosa','versicolor','virginica']
merker_iris =['o','s','x']
for i in range(0,3):
    plt.scatter(x=iris.Petal_Width[iris['Species'] ==sepcies[i]], y=iris.Petal_Length[iris['Species'] == sepcies[i]], s=20,
                color=colors_iris[i], marker=merker_iris[i], label=sepcies[i])
plt.xlabel('花瓣宽度',fontsize =12, labelpad =20)
plt.ylabel('花瓣长度',fontsize = 12,  labelpad =20)
plt.title('不同种类的鸢尾花花瓣宽度和长度关系图',fontsize =12)
plt.legend(loc='upper left')
plt.show()

# # # # # # 折线图
pd.set_option('display.max_columns', 8)
data = np.load('国民经济核算季度数据.npz')
name = data['columns'] ## 提取其中的columns数组，视为数据的标签
values = data['values']## 提取其中的values数组，数据的存在位置
## 绘制折线图
fig = plt.figure(figsize=(16,9)) # 创建画布
ax = fig.add_axes([0.15,0.2,0.8,0.7]) # Axes是画布上的绘图区域，可以添加多块
plt.plot(values[:,0],values[:,2],color = 'r',linestyle = '--')
plt.xlabel('年份',labelpad=20)## 添加横轴标签
plt.ylabel('生产总值（亿元）')## 添加y轴名称
plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
plt.title('2000-2017年季度生产总值折线图')## 添加图表标题
plt.savefig('2000-2017年季度生产总值折线图.pdf')
plt.show()

# # # 不同类别
fig =plt.figure(figsize=(8,7)) # 创建画布
ax = fig.add_axes([0.15,0.2,0.8,0.7]) # Axes是画布上的绘图区域，可以添加多块
plt.plot(values[:,0],values[:,3],'bs-',
       values[:,0],values[:,4],'ro-.',
       values[:,0],values[:,5],'gH--')## 绘制折线图
plt.xlabel('年份')## 添加横轴标签
plt.ylabel('生产总值（亿元）')## 添加y轴名称
plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
plt.title('2000-2017年各产业季度生产总值折线图')## 添加图表标题
plt.legend(['第一产业','第二产业','第三产业'])
# plt.savefig('2000-2017年季度各产业生产总值折线图.pdf')
plt.show()

# 绘制子图
p1 = plt.figure(figsize=(8,7))## 设置画布
## 子图1
ax3 = p1.add_subplot(2,1,1)
plt.plot(values[:,0],values[:,3],'b-',
        values[:,0],values[:,4],'r-.',
        values[:,0],values[:,5],'g--')## 绘制折线图
plt.ylabel('生产总值（亿元）')## 添加纵轴标签
plt.title('2000-2017年各产业季度生产总值折线图')## 添加图表标题
plt.legend(['第一产业','第二产业','第三产业'])## 添加图例
## 子图2
ax4 = p1.add_subplot(2,1,2)
plt.plot(values[:,0],values[:,6], 'r-',## 绘制折线图
        values[:,0],values[:,7], 'b-.',## 绘制折线图
        values[:,0],values[:,8],'y--',## 绘制折线图
        values[:,0],values[:,9], 'g:',## 绘制折线图
        values[:,0],values[:,10], 'c-',## 绘制折线图
        values[:,0],values[:,11], 'm-.',## 绘制折线图
        values[:,0],values[:,12], 'k--',## 绘制折线图
        values[:,0],values[:,13], 'r:',## 绘制折线图
        values[:,0],values[:,14], 'b-')## 绘制折线图
plt.legend(['农业','工业','建筑','批发','交通',
        '餐饮','金融','房地产','其他'])
plt.xlabel('年份')## 添加横轴标签
plt.ylabel('生产总值（亿元）')## 添加纵轴标签
plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
plt.show()

#==============================================================================
# 2.2.2 误差棒图
#==============================================================================
# # #散点图的误差棒图
x = np.linspace(0.1,0.6,10)
y = np.exp(x)
error = 0.05 + 0.15*x
lower_error = error
upper_error = 0.3*error
error_limit = [lower_error,upper_error]
plt.errorbar(x,y,yerr=error_limit,fmt=':o',
             ecolor='y',elinewidth=4,ms=5,mfc='c',mec='r',capthick=5,capsize=5)
plt.xlim(0,0.7)
plt.savefig('误差棒.pdf')
plt.show()


# # #柱状图的误差棒图
GDP = pd.read_excel('Province GDP 2017.xlsx')
std_err = [0.2,0.45,0.3,0.26,0.5,0.3]
error_attri = dict(elinewidth=2,ecolor='black',capsize=3)
colors= ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#00FFFF']
fig =plt.figure(figsize=(8,7)) # 创建画布
plt.bar(GDP.index.values,GDP.GDP,width=0.6,align='center',yerr=std_err,error_kw=error_attri,color= colors,
        tick_label=GDP.Province)
plt.xticks(rotation=45)
plt.xlabel('省份',fontsize = 20,labelpad =10)
plt.ylabel('GDP产值(万亿)',fontsize = 20,labelpad =20)
plt.grid(True,axis='y',ls=':',color='gray',alpha=0.8)
plt.title('2017年6个省份的GDP',fontsize = 25)
plt.show()

# # #条形图的误差棒图
bar_width =0.6
colors= ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#00FFFF']
std_err = [0.2,0.45,0.3,0.26,0.5,0.3]
error_attri = dict(elinewidth=2,ecolor='black',capsize=3)
plt.barh(y =GDP.index.values,width = GDP.GDP,height=bar_width,align='center',xerr=std_err,color = colors,
        tick_label=GDP.Province)
plt.xlabel('GDP产值(万亿)',fontsize = 20,labelpad =20)
plt.ylabel('省份',fontsize = 20,labelpad =20)
plt.grid(True,axis='x',ls=':',color='darkorange',alpha=0.8)
plt.title('2017年6个省份的GDP',fontsize = 25)
plt.show()


#==============================================================================
# 2.2.3 完善统计图形
#==============================================================================
# # # # # # 图例和标题以及画图使用
data = np.load('国民经济核算季度数据.npz')
name = data['columns'] ## 提取其中的columns数组，视为数据的标签
values = data['values']## 提取其中的values数组，数据的存在位置
fig =plt.figure(figsize=(8,7)) # 创建画布
ax = fig.add_axes([0.10,0.2,0.85,0.7]) # Axes是画布上的绘图区域，可以添加多块
plt.plot(values[:,0],values[:,3],'bs-',label='第一产业')
plt.plot(values[:,0],values[:,4],'ro-.',label='第二产业')
plt.plot(values[:,0],values[:,5],'gH--',label='第三产业')## 绘制折线图
plt.xlabel('年份',labelpad=15,fontsize=20)## 添加横轴标签
plt.ylabel('生产总值（亿元）',labelpad=15,fontsize=20,style='oblique')## 添加y轴名称
plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
plt.title('2000-2017年各产业季度生产总值折线图',fontsize=20)## 添加图表标题
plt.legend(loc='upper right',bbox_to_anchor=(0.10,0.95),ncol=1, frameon=True, #是否要边框
           title ='不同产业的比较',shadow=False, fancybox=False)
# plt.savefig('2000-2017年季度各产业生产总值折线图.pdf')
plt.show()

# # # # # # 调整刻度轴
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
         markerfacecolor='brown') # 点的填充色
# 添加y轴标签
plt.ylabel('人数')
# 添加图形标题
plt.title('每天微信文章阅读人数趋势')
# 显示图形
plt.show()

# 绘制阅读人数折线图
plt.plot(wechat.Date, # x轴数据
         wechat.Counts, # y轴数据
         linestyle = '-', # 折线类型，实心线
         color = 'steelblue', # 折线颜色
         label = '阅读人数')
# 绘制阅读人次折线图
plt.plot(wechat.Date, # x轴数据
         wechat.Times, # y轴数据
         linestyle = '--', # 折线类型，虚线
         color = 'indianred', # 折线颜色
         label = '阅读人次')
import matplotlib as mpl
# 获取图的坐标信息
ax = plt.gca()
# 设置日期的显示格式
date_format = mpl.dates.DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_format)
# 设置x轴显示多少个日期刻度
# xlocator = mpl.ticker.LinearLocator(10)
# 设置x轴每个刻度的间隔天数
xlocator = mpl.ticker.MultipleLocator(6)
ax.xaxis.set_major_locator(xlocator)
# 为了避免x轴刻度标签的紧凑，将刻度标签旋转45度
plt.xticks(rotation=45)
# 添加y轴标签
plt.ylabel('人数')
# 添加图形标题
plt.title('每天微信文章阅读人数与人次趋势')
# 添加图例
plt.legend()
# 显示图形
plt.show()

# # # 货币和时间序列样式的刻度标签
from calendar import month_name, day_name
from matplotlib.ticker import  FormatStrFormatter
fig = plt.figure()
ax = fig.add_axes([0.2,0.2,0.7,0.7])
x = np.arange(1,8,1)
y = 2*x
ax.plot(x,y,ls='-',lw=2,color='orange',marker='o',ms=10,mfc='c',mec='c')
ax.yaxis.set_major_formatter(FormatStrFormatter('r$\yen%1.1f$'))
plt.xticks(x,day_name[0:7],rotation=20)
ax.set_xlim(0,8)
ax.set_ylim(1,18)
plt.show()

# 逆序坐标轴
time = np.arange(1,11,0.5)
machinepower = np.power(time,2) + 0.8
fig = plt.figure(figsize=(10,8))
plt.plot(time,machinepower,
         linestyle = '-', # 折线类型
         linewidth = 2, # 折线宽度
         color = 'steelblue', # 折线颜色
         marker = 'o', # 折线图中添加圆点
         markersize = 6, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='brown') # 点的填充色
plt.xlim(10,1)
plt.xlabel('使用年限')
plt.ylabel('机器功率')
plt.title('机器损耗曲线')
plt.grid(ls=":",lw=2,color='gray',alpha=0.5)
plt.show()

# # # # # # 向图形添加统计表格
labels =["A难度水平",'B难度水平','C难度水平','D难度水平']
students = [0.35,0.15,0.20,0.30]
colors = ['red','green','blue','yellow']
explode = (0.1,0.1,0,0)
plt.pie(students,explode = explode,labels =labels,autopct='%1.1f%%',startangle=45,shadow=True,
        colors=colors)
# 设置x，y轴刻度一致，这样饼图才能是圆的
plt.axis('equal')
plt.title('选择不同难度测试试卷的学生百分比')
# 添加表格
col_labels = ["A难度水平",'B难度水平','C难度水平','D难度水平']
row_labels = ['学生选择试卷人数']
table_vals =np.array([3500,1500,2000,3000]).reshape(1,-1)
col_colors = ['red','green','blue','yellow']
my_table = plt.table(cellText=table_vals,cellLoc='center' ,colWidths=[0.1] * 4,
   rowLabels=row_labels, colLabels=col_labels,colColours=col_colors, rowLoc='center',loc='bottom')
plt.show()

# # #柱状图中插入表格
GDP = pd.read_excel('Province GDP 2017.xlsx')
colors= ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#00FFFF']
fig =plt.figure(figsize=(8,7)) # 创建画布
plt.bar(GDP.index.values,GDP.GDP,width=0.6,align='center',color= colors,
        tick_label=GDP.Province)
plt.xticks(rotation=45)
plt.xlabel('省份',fontsize = 20,labelpad =15)
plt.ylabel('GDP产值(万亿)',fontsize = 20,labelpad =15)
plt.grid(True,axis='y',ls=':',color='gray',alpha=0.8)
plt.title('2017年6个省份的GDP',fontsize = 25)
# 添加表格
col_labels = ['GDP(万亿)']
row_labels = GDP.Province
table_vals =np.array(GDP.GDP.values).reshape(-1,1)
col_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#00FFFF']
my_table = plt.table(cellText=table_vals,cellLoc='center' ,colWidths=[0.1] * 6,
   rowLabels=row_labels, colLabels=col_labels,rowColours=col_colors,bbox=[0.8,0.7,0.1,0.25])
plt.show()


