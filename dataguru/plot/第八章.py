"""
这是第8章的知识点
"""
# 第八章  章 Python高级绘图三实现共享坐标轴
# 导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.chdir('C:\data\数据')
pd.set_option('display.max_columns',8)
#==============================================================================
# 8.1 ggplot
#==============================================================================
# 使用ggplot风格
GDP = pd.read_excel('Province GDP 2017.xlsx')
#plt.style.use('ggplot')  # 绘图风格使用ggplot
plt.bar(x=GDP.index.values,height =GDP.GDP,tick_label=GDP.Province,
        color = 'steelblue')
plt.ylabel('GDP(万亿',fontsize=20)
plt.title('2017年6个省份的GDP',fontsize = 25)
plt.show()


#plt.style.use('default') #还原,记得还原后，重新运行一遍中文设置
print(plt.style.available) #绘图风格
print(len(plt.style.available))
# 尝试将style改为以上种类，再分别运行
# 画分布图
# 导入数据
Titanic = pd.read_csv('titanic_train.csv')
#检查年龄是否有缺失
any(Titanic['Age'].isnull())
# 删除缺失值
Titanic['Age'].dropna(inplace=True)
#绘制直方图
plt.style.use('classic') # 使用classic
plt.hist(x =Titanic.Age,bins=30,color='r',edgecolor='black', rwidth=1) # rwidth柱状宽度
plt.xlabel('年龄')
plt.ylabel('频数')
plt.title('年龄分布图')
plt.show()

#==============================================================================
# 8.2 seaborn绘图
#==============================================================================
# # # # # # seaborn基本用法
# 调用seaborn第一种方法
plt.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.bar(x=GDP.index.values,height =GDP.GDP,tick_label=GDP.Province,
        color = 'steelblue')
plt.ylabel('GDP(万亿',fontsize=20)
plt.title('2017年6个省份的GDP',fontsize = 25)
plt.show()

# 调用seaborn第二种方法
import seaborn as sns
sns.set(style='darkgrid',context='notebook',font_scale=1.5) # 设置背景
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.bar(x=GDP.index.values,height =GDP.GDP,tick_label=GDP.Province,
        color = 'steelblue',align='center')
plt.ylabel('GDP(万亿',fontsize=20)
plt.title('2017年6个省份的GDP',fontsize = 25)
plt.show()

# 调用seaborn第三种方法
sns.barplot(x = 'Province',y = 'GDP',data=GDP,color='steelblue',
            orient ='vertical')
plt.ylabel('GDP(万亿',fontsize=20)
plt.title('2017年6个省份的GDP',fontsize = 25)
plt.ylim(0,10)
for x,y in enumerate(GDP.GDP):
    plt.text(x,y+0.1,'%s万亿'  %round(y,1),ha='center',fontsize =18)
plt.show()

# 调整seaborn的设置
# style:风格 (darkgrid whitegrid dark white ticks)
# context:线条粗细 paper，notebook, talk, and poster
# palette :调色板 deep, muted, pastel, bright, dark, colorblind
# font: 字体 一般用默认，不调整
# font_scale 控制坐标轴刻度大小
sns.set(style='darkgrid',context='notebook',palette ='deep',font_scale=1.5) # 设置背景
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.barplot(x = 'Province',y = 'GDP',data=GDP,color='steelblue',
            orient ='vertical')
plt.ylabel('GDP(万亿)',fontsize=20)
plt.title('2017年6个省份的GDP',fontsize = 25)
plt.ylim(0,10)
for x,y in enumerate(GDP.GDP):
    plt.text(x,y+0.1,'%s万亿'  %round(y,1),ha='center',fontsize =18)
plt.show()

# # # # # # 绘制统计图形
# # #条形图
sns.set(style='darkgrid',context='notebook',palette ='deep',font_scale=1.5) # 设置背景
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.barplot(y = 'Province',x = 'GDP',data=GDP,color='steelblue',
            orient ='horizontal')
plt.xlabel('GDP(万亿)',fontsize=20)
plt.ylabel('')
plt.title('2017年6个省份的GDP',fontsize = 25)
plt.xlim(0,10)
for y,x in enumerate(GDP.GDP):
    plt.text(x+0.1,y,'%s' %round(x,2),va='center')
    plt.text(x,y+0.2,'%s' %'(万亿)',va='center')
plt.show()

# # #按照分类变量统计
Industry_GDP = pd.read_excel('Industry_GDP.xlsx')
quarter  =['第一季度','第二季度','第三季度','第四季度']
sns.barplot(x = 'Quarter',y = 'GDP',hue='Industry_Type',data=Industry_GDP,palette="husl",
            orient ='vertical')
plt.ylabel('GDP(万亿)',fontsize=20)
plt.xlabel('')
plt.xticks(np.arange(4),quarter,fontsize=12)
plt.title('2017年四个季度GDP情况',fontsize = 25)
plt.show()

# 分类变量，条形图
quarter  =['第一季度','第二季度','第三季度','第四季度']
sns.barplot(x = 'GDP',y = 'Quarter',hue='Industry_Type',data=Industry_GDP,palette="husl",
            orient ='horizontal')
plt.xlabel('GDP(万亿)',fontsize=20)
plt.ylabel('')
plt.yticks(np.arange(4),quarter,fontsize=12)
plt.title('2017年四个季度GDP情况',fontsize = 25)
plt.legend(bbox_to_anchor=(1.12,0.85),ncol=1, frameon=False, #是否要边框
       shadow=False, fancybox=True,fontsize=12)
plt.show()

# # # # # # 散点图
# # # 绘制简单散点图
iris = pd.read_csv('iris.csv')
sns.scatterplot(x = 'Petal_Width',y = 'Petal_Length',data = iris,color="red", marker='+', s=20)
plt.xlabel('花瓣宽度')
plt.ylabel('花瓣长度')
plt.title('鸢尾花花瓣宽度和长度关系图')
plt.show()

# 绘制分类特征下面的散点图
# plt.scatter(x = iris.Petal_Width[iris['Species'] =='setosa'],y = iris.Petal_Length[iris['Species'] =='setosa'],s =20,
#             color ='steelblue',marker='o',label = 'setosa',alpha=0.8)
# plt.scatter(x = iris.Petal_Width[iris['Species'] =='versicolor'],y = iris.Petal_Length[iris['Species'] =='versicolor'],s =30,
#             color ='indianred',marker='s',label = 'versicolor')
# plt.scatter(x = iris.Petal_Width[iris['Species'] =='virginica'],y = iris.Petal_Length[iris['Species'] =='virginica'],s =40,
#             color ='green',marker='x',label = 'virginica')
# plt.xlabel('花瓣宽度',fontsize=12)
# plt.ylabel('花瓣长度')
# plt.title('不同种类的鸢尾花花瓣宽度和长度关系图')
# plt.legend(loc='upper left')
# plt.show()
# muted,RdBu,Set1,Blues_d ,husl
plt.figure(figsize=(6,6))
sns.scatterplot(x = 'Petal_Width',y = 'Petal_Length',hue='Species',data =iris,style ='Species',s=100,
                palette='muted')
plt.xlabel('花瓣宽度')
plt.ylabel('花瓣长度')
plt.title('不同种类鸢尾花花瓣宽度和长度关系图')
plt.show()

sns.lmplot(x = 'Petal_Width',y = 'Petal_Length',hue='Species',data =iris,legend_out=False,\
           truncate=False,markers=['o','D','x'],fit_reg=False,aspect=1.8,height=7,
           scatter_kws={'s':100,'facecolor':'red'})
plt.xlabel('花瓣宽度')
plt.ylabel('花瓣长度')
plt.title('不同种类鸢尾花花瓣宽度和长度关系图',fontsize=30)
plt.show()

# # # # # #箱线图
# 绘制单个变量箱线图
Titanic = pd.read_csv('titanic_train.csv')
# 删除缺失值
Titanic['Age'].dropna(inplace=True)
sns.boxplot( y = 'Age', data = Titanic,
             showmeans=True,color = 'steelblue',width = 0.3, linewidth=2,
            flierprops = {'marker':'o','markerfacecolor':'red', 'markersize':3},
            meanprops = {'marker':'D','markerfacecolor':'indianred', 'markersize':10},
            medianprops = {'linestyle':'--','color':'orange'})
# 更改x轴和y轴标签
plt.xlabel('')
plt.ylabel('年龄')
# 添加标题
plt.title('年龄箱线分布图')
# 显示图形
plt.show()

# 绘制分类变量箱线图
pd.set_option('display.max_columns',8)
sec_building = pd.read_excel('sec_buildings.xlsx')
group_region = sec_building.groupby('region')
avg_price = group_region.aggregate({'price_unit':np.mean}).sort_values('price_unit', ascending = False)
sns.boxplot(x = 'region', y = 'price_unit', data = sec_building ,
            order = avg_price.index, showmeans=True,color = 'steelblue',
            flierprops = {'marker':'o','markerfacecolor':'red', 'markersize':3},
            meanprops = {'marker':'D','markerfacecolor':'indianred', 'markersize':4},
            medianprops = {'linestyle':'--','color':'orange'})
# 更改x轴和y轴标签
plt.xlabel('')
plt.ylabel('单价（元）')
# 添加标题
plt.title('不同行政区域的二手房单价对比')
# 显示图形
plt.show()

# # # # # # 绘制直方图
# seaborn模块绘制分组的直方图和核密度图
# 取出男性年龄
Age_Male = Titanic.Age[Titanic.Sex == 'male']
# 取出女性年龄
Age_Female = Titanic.Age[Titanic.Sex == 'female']

# 绘制男女乘客年龄的直方图
from scipy.stats import norm
import seaborn as sns
sns.set(style='darkgrid',context='notebook',font_scale=1.5) # 设置背景
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.distplot(Age_Male, bins = 30, kde = False, hist_kws = {'color':'steelblue'}, norm_hist=True, label = '男性')
sns.distplot(Age_Male, hist = False, kde=False,fit = norm,fit_kws = {'color':'yellow', 'linestyle':'-'},
             norm_hist = True, label = '男性年龄正态分布图')
# 绘制女性年龄的直方图
sns.distplot(Age_Female, bins = 30, kde = False, hist_kws = {'color':'purple'}, label = '女性',norm_hist=True)
sns.distplot(Age_Female, hist = False, kde=False,fit=norm,fit_kws = {'color':'blue', 'linestyle':'--'},
             norm_hist = True, label = '女性年龄正态分布图')
plt.title('男女乘客的年龄直方图')
# 显示图例
plt.legend()
# 显示图形
plt.show()

# 绘制男女乘客年龄的核密度图
sns.distplot(Age_Male, hist = False, kde_kws = {'color':'red', 'linestyle':'-'},
             norm_hist = True, label = '男性核密度图')
sns.distplot(Age_Male, hist = False, kde=False,fit=norm,fit_kws = {'color':'yellow', 'linestyle':'-'},
              norm_hist = True, label = '男性概率密度图')
# 绘制女性年龄的核密度图
sns.distplot(Age_Female, hist = False, kde_kws = {'color':'black', 'linestyle':'--'},
             norm_hist = True, label = '女性核密度图')
sns.distplot(Age_Female, hist = False, kde=False,fit=norm,fit_kws = {'color':'blue', 'linestyle':'--'},
             norm_hist = True, label = '女性概率密度图')
plt.title('男女乘客的年龄核密度图')
# 显示图例
plt.legend()
# 显示图形
plt.show()

# # # # # # 折线图
sns.set(style='ticks',context='notebook') # 设置背景
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
jd_stock  = pd.read_csv('data.csv', sep =',',header=None,names =['name','date','opening_price','closing_price',
                                                                 'lowest_price','highest_price','volume'])
sns.lineplot(x="date", y="opening_price",data =jd_stock,lw=2,color="red")
# 添加y轴标签
plt.ylabel('开盘价')
plt.xlabel('')
plt.xticks(range(0,71,4),jd_stock.loc[range(0,71,4),'date'],rotation=45,fontsize=12)
# 添加图形标题
plt.title('开盘价变化趋势')
# 显示图形
plt.show()

# # # # # # 综合案例(非常重要)
# # 读取数据
Prod_Trade = pd.read_excel('Prod_Trade.xlsx')
# 衍生出交易年份和月份字段
Prod_Trade['year'] = Prod_Trade.Date.dt.year
Prod_Trade['month'] = Prod_Trade.Date.dt.month
# 设置大图框的长和高
plt.figure(figsize = (12,6))
# 设置第一个子图的布局
ax1 = plt.subplot2grid(shape = (2,3), loc = (0,0))
# 统计2012年各订单等级的数量
Class_Counts = Prod_Trade.Order_Class[Prod_Trade.year == 2012].value_counts()
Class_Percent = Class_Counts/Class_Counts.sum()
# 将饼图设置为圆形（否则有点像椭圆）
ax1.set_aspect(aspect = 'equal')
# 绘制订单等级饼图
ax1.pie(x = Class_Percent.values, labels = Class_Percent.index, autopct = '%.1f%%')
# 添加标题
ax1.set_title('各等级订单比例')

# 设置第二个子图的布局
ax2 = plt.subplot2grid(shape = (2,3), loc = (0,1))
# 统计2012年每月销售额
Month_Sales = Prod_Trade[Prod_Trade.year == 2012].groupby(by = 'month').aggregate({'Sales':np.sum})
# 绘制销售额趋势图
Month_Sales.plot(title = '2012年各月销售趋势', ax = ax2, legend = False)
# 删除x轴标签
ax2.set_xlabel('')
# 设置第三个子图的布局
ax3 = plt.subplot2grid(shape = (2,3), loc = (0,2), rowspan = 2)
# 绘制各运输方式的成本箱线图
sns.boxplot(x = 'Transport', y = 'Trans_Cost', data = Prod_Trade, ax = ax3)
# 添加标题
ax3.set_title('各运输方式成本分布')
# 删除x轴标签
ax3.set_xlabel('')
# 修改y轴标签
ax3.set_ylabel('运输成本')
# 设置第四个子图的布局
ax4 = plt.subplot2grid(shape = (2,3), loc = (1,0), colspan = 2)
# 2012年客单价分布直方图
sns.distplot(Prod_Trade.loc[Prod_Trade.year == 2012,'Sales'], bins = 40, norm_hist = True,
             ax = ax4, hist_kws = {'color':'steelblue'}, kde_kws=({'linestyle':'--', 'color':'red'}))
# 添加标题
ax4.set_title('2012年客单价分布图')
# 修改x轴标签
ax4.set_xlabel('销售额')
# 调整子图之间的水平间距和高度间距
plt.subplots_adjust(hspace=0.6, wspace=0.3)
# 图形显示
plt.show()
