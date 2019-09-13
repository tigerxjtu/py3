"""
这是第二章第4节的知识点
Python数据可视化
"""
# 2.4 python精美作图
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
#==============================================================================
# 2.4.1 ggplot绘图
#==============================================================================
# 使用ggplot风格
GDP = pd.read_excel('Province GDP 2017.xlsx')
mpl.style.use('ggplot')  # 绘图风格使用ggplot
plt.bar(x=GDP.index.values,height =GDP.GDP,tick_label=GDP.Province,
        color = 'steelblue')
plt.ylabel('GDP(万亿',fontsize=20)
plt.title('2017年6个省份的GDP',fontsize = 25)
plt.show()

#mpl.style.use('default') #还原,记得还原后，重新运行一遍中文设置
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
mpl.style.use('classic') # 使用classic
plt.hist(x =Titanic.Age,bins=30,color='r',edgecolor='black', rwidth=1) # rwidth柱状宽度
plt.xlabel('年龄')
plt.ylabel('频数')
plt.title('年龄分布图')
plt.show()

#==============================================================================
# 2.4.2  seaborn绘图
#==============================================================================
# # # # # # seaborn基本用法
# 调用seaborn第一种方法
mpl.style.use('seaborn')
plt.bar(x=GDP.index.values,height =GDP.GDP,tick_label=GDP.Province,
        color = 'steelblue')
plt.ylabel('GDP(万亿',fontsize=20)
plt.title('2017年6个省份的GDP',fontsize = 25)
plt.show()
# 调用seaborn第二种方法
import seaborn as sns
sns.set(style='darkgrid',context='notebook',font_scale=1.5) # 设置背景
# 支持中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
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
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
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
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
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
sns.barplot(x = 'Quarter',y = 'GDP',hue='Industry_Type',data=Industry_GDP,color='blue',palette="husl",
            orient ='vertical')
plt.ylabel('GDP(万亿)',fontsize=20)
plt.xlabel('')
plt.xticks(np.arange(4),quarter,fontsize=12)
plt.title('2017年四个季度GDP情况',fontsize = 25)
plt.show()

# 分类变量，条形图
quarter  =['第一季度','第二季度','第三季度','第四季度']
sns.barplot(x = 'GDP',y = 'Quarter',hue='Industry_Type',data=Industry_GDP,color='blue',palette="husl",
            orient ='horizontal')
plt.xlabel('GDP(万亿)',fontsize=20)
plt.ylabel('')
plt.yticks(np.arange(4),quarter,fontsize=12)
plt.title('2017年四个季度GDP情况',fontsize = 25)
plt.legend(loc='center right',bbox_to_anchor=(1.12,0.85),ncol=1, frameon=False, #是否要边框
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
# #             color ='steelblue',marker='o',label = 'setosa',alpha=0.8)
# # plt.scatter(x = iris.Petal_Width[iris['Species'] =='versicolor'],y = iris.Petal_Length[iris['Species'] =='versicolor'],s =30,
# #             color ='indianred',marker='s',label = 'versicolor')
# # plt.scatter(x = iris.Petal_Width[iris['Species'] =='virginica'],y = iris.Petal_Length[iris['Species'] =='virginica'],s =40,
# #             color ='green',marker='x',label = 'virginica')
# # plt.xlabel('花瓣宽度',fontsize=12)
# # plt.ylabel('花瓣长度')
# # plt.title('不同种类的鸢尾花花瓣宽度和长度关系图')
# # plt.legend(loc='upper left')
# # plt.show()
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
             showmeans=True,color = 'steelblue',width =0.3, linewidth=2,
            flierprops = {'marker':'o','markerfacecolor':'red', 'markersize':3},
            meanprops = {'marker':'D','markerfacecolor':'indianred', 'markersize':4},
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
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
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
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
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
# 读取数据
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

#==============================================================================
# 2.4.3  pyechart
#==============================================================================
# # # # # # 基本用法
from  pyecharts import Pie
# # # # # #绘制柱状图
from pyecharts import Bar
GDP = pd.read_excel('Province GDP 2017.xlsx')
province = GDP.Province
GDP_value = GDP.GDP
bar =Bar("2017年6个省份GDP情况", "单位(万亿)",title_pos='left',width=800) #创建一个实例对象
bar.add("GDP",GDP.Province, GDP.GDP,is_legend_show =True ) #是否显示顶端图例
bar.show_config()
bar.render('GDP.html')

# 条形图
bar =Bar("2017年6个省份GDP情况", "单位(万亿)",title_pos='left',width=800) #创建一个实例对象
bar.add("GDP",GDP.Province,GDP.GDP,is_convert=True)
# bar.show_config()
bar.render('GDP.html')

# # # # # #饼图
# radius 控制饼图的半径
attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "鞋", "袜子"]
v1 = [110, 112, 133, 100, 108, 120]
pie = Pie("服装占比",title_pos='left',width=800)
pie.add("", attr, v1, is_label_show=True,radius=[30,75])
pie.render('pie.html')

#label_text_color字体颜色
attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "鞋", "袜子"]
v1 = [110, 112, 133, 100, 108, 120]
pie = Pie("饼图-服装占比", title_pos='center')
pie.add("服饰",attr,v1,radius=[40, 75],
    label_text_color='red',is_label_show=True,legend_orient="vertical",legend_pos="left")
pie.render('pie.html')

# # # # # #散点图
from pyecharts import Scatter
iris = pd.read_csv('iris.csv')
scatter = Scatter('鸢尾花花瓣宽度和长度关系图')
scatter.add("setosa", iris.Petal_Width[iris.Species=='setosa'], iris.Petal_Length[iris.Species=='setosa'],
            symbol_size = 8,mark_point_symbol='arrow')
scatter.add("versicolor", iris.Petal_Width[iris.Species=='versicolor'], iris.Petal_Length[iris.Species=='versicolor'],
            symbol_size =10)
scatter.add("virginica", iris.Petal_Width[iris.Species=='virginica'], iris.Petal_Length[iris.Species=='virginica'],
            symbol_size =12)
scatter.render('iris.html')

# # # # # #折线图
# 读取数据
GDP_data = pd.read_excel('国民经济核算季度数据.xlsx')
# 选择部分数据
# mark_point_symbol 指定标签的形状
# mark_point_textcolor 指定标签的字体颜色
# mark_point_symbolsize 指定标记标记的大小
# mark_line 指定标记线
GDP_data = GDP_data.loc[30:68,:]
from pyecharts import Line
line = Line('GDP和工业增加值变化趋势',title_pos='left')
line.add('国内生产总值',GDP_data.时间,GDP_data['国内生产总值_当季值(亿元)'],mark_point=['average','max','min'],
         mark_point_symbol='diamond',mark_point_textcolor='#40ff27')
line.add('工业增加值',GDP_data.时间,GDP_data['工业增加值_当季值(亿元)'],mark_point=['max'], is_smpooth=True, mark_line=['average','max','min'],
         mark_point_symbol='arrow',mark_point_symbolsize=40)
line.render('GDP.data.html')


# # # 柱状图和折线图结合
from pyecharts import Bar, Line,Overlap #导入相关模块
line = Line('') #创建一个实例对象
line.add('国内生产总值',GDP_data.时间,GDP_data['国内生产总值_当季值(亿元)'],mark_point=['average','max','min'],
         mark_point_symbol='diamond',mark_point_textcolor ='#40ff27')
bar = Bar('GDP变化趋势')
bar.add('工业增加值',GDP_data['时间'],GDP_data['工业增加值_当季值(亿元)'])
overlap = Overlap() #
overlap.add(line)
overlap.add(bar)
overlap.render('GDPdata.html')
# # # # # # 仪表盘
from pyecharts import Gauge
gauge =Gauge('目标完成率')
gauge.add('任务指标','完成率',90)
gauge.render('目标完成率.html')

# # # # # # 箱线图
Titanic = pd.read_csv('titanic_train.csv')
#检查年龄是否有缺失
any(Titanic['Age'].isnull())
# 删除缺失值
Titanic['Age'].dropna(inplace=True)
from pyecharts import Boxplot
boxplot = Boxplot('年龄箱线图')
x_axis = ['年龄']
y_axis = Titanic['Age'].values
y_axis = list(np.reshape(y_axis,(1,-1)))
_yaxis = boxplot.prepare_data(y_axis) #必须要将数据进行转换
boxplot.add('箱线图',x_axis,_yaxis)
boxplot.render('boxplot.html')


# # # # # #补充，不要求学会
# # #子图
from pyecharts import Line, Pie, Grid
line = Line("折线图示例", width=1200)
attr = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
line.add("最高气温", attr, [11, 11, 15, 13, 12, 13, 10],
         mark_point=["max", "min"], mark_line=["average"])
line.add("最低气温", attr, [1, -2, 2, 5, 3, 2, 0], mark_point=["max", "min"],
         mark_line=["average"], legend_pos="20%")
attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
v1 = [11, 12, 13, 10, 10, 10]
pie = Pie("饼图示例", title_pos="45%")
pie.add("", attr, v1, radius=[30, 55],
        legend_pos="65%", legend_orient='vertical')
grid = Grid()
grid.add(line, grid_right="70%")
grid.add(pie, grid_left="40%")
grid.render('12.html')

# # # # # #3D 散点图
from pyecharts import Scatter3D
iris = pd.read_csv('iris.csv')
scatter3d = Scatter3D('鸢尾花花瓣宽度和长度关系图',)
setosa = pd.DataFrame({'petal_width':iris.Petal_Width[iris.Species=='setosa'].values,
                      'Petal_Length':iris.Petal_Length[iris.Species=='setosa'].values,
                      'iris.Sepal_Width':iris.Sepal_Width[iris.Species == 'setosa'].values})
setosa = list(setosa.values)

versicolor = pd.DataFrame({'petal_width':iris.Petal_Width[iris.Species=='versicolor'].values,
                      'Petal_Length':iris.Petal_Length[iris.Species=='versicolor'].values,
                      'iris.Sepal_Width':iris.Sepal_Width[iris.Species == 'versicolor'].values})
versicolor= list(versicolor.values)

virginica = pd.DataFrame({'petal_width':iris.Petal_Width[iris.Species=='virginica'].values,
                      'Petal_Length':iris.Petal_Length[iris.Species=='virginica'].values,
                      'iris.Sepal_Width':iris.Sepal_Width[iris.Species == 'virginica'].values})
virginica= list(virginica.values)
scatter3d.add("setosa", setosa, symbol_size =8,mark_point_symbol='arrow',color='red')
scatter3d.add("versicolor", versicolor,symbol_size =10)
scatter3d.add("virginica", virginica, symbol_size =12)
scatter3d.render('iris.html')


# # # 散点图
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
iris = pd.read_csv('iris.csv')
#  将数据点分成三部分画，在颜色上有区分度
# ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
fig= plt.figure()
ax = Axes3D(fig)  # 创建一个三维的绘图工程
ax.scatter(iris.Petal_Width[iris.Species=='setosa'], iris.Petal_Length[iris.Species=='setosa'],
           iris.Sepal_Width[iris.Species=='setosa'],c='y',marker='+',label ='setosa')
ax.scatter(iris.Petal_Width[iris.Species=='versicolor'], iris.Petal_Length[iris.Species=='versicolor'],
           iris.Sepal_Width[iris.Species=='versicolor'],c='r',marker='o',label='versicolor')
ax.scatter(iris.Petal_Width[iris.Species=='virginica'], iris.Petal_Length[iris.Species=='virginica'],
           iris.Sepal_Width[iris.Species=='virginica'],c='b',label='virginica')
ax.set_zlabel('Sepal_Width')  # 坐标轴
ax.set_ylabel('Petal_Length')
ax.set_xlabel('Petal_Width')
ax.set_xlim(0,5)
ax.set_ylim(0, 7)
ax.set_zlim(0, 3)
plt.legend()
ax.view_init(elev=20., azim=-15)
plt.show()













