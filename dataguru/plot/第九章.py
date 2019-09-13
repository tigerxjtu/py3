#导入相关库
import  pandas as pd
import numpy as np
import os
os.chdir(r'D:\炼数成金数据可视化\数据')
#==============================================================================
# 9 pyecharts
#==============================================================================
# # # # # # 基本用法
from  pyecharts import Pie
# # # # # #绘制柱状图
from pyecharts import Bar

import os
os.chdir('c:\data')

GDP = pd.read_excel('Province GDP 2017.xlsx')
province = GDP.Province
GDP_value = GDP.GDP
bar =Bar("2017年6个省份GDP情况", "单位(万亿)",title_pos='left',width=800) #创建一个实例对象
bar.add("GDP",GDP.Province, GDP.GDP,is_legend_show =True ) #是否显示顶端图例
#bar.show_config()
bar.render('GDP.html')

# 条形图
bar =Bar("2017年6个省份GDP情况", "单位(万亿)",title_pos='left',width=800) #创建一个实例对象
bar.add("GDP",GDP.Province,GDP.GDP,is_convert=True)
# bar.show_config()
bar.render('GDP.html')

# # # # # #饼图
from  pyecharts import Pie
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
            symbol_size = 8, mark_point_symbol='arrow')
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
GDP_data = GDP_data.loc[30:68]
from pyecharts import Line
line = Line('GDP和工业增加值变化趋势',title_pos='left')
line.add('国内生产总值',GDP_data.时间,GDP_data['国内生产总值_当季值(亿元)'],mark_point=['average','max','min'],
         mark_point_symbol='diamond',mark_point_textcolor='blue')
line.add('工业增加值',GDP_data.时间,GDP_data['工业增加值_当季值(亿元)'],mark_point=['max'], is_smpooth=True, mark_line=['average','max','min'],
         mark_point_symbol='arrow',mark_point_symbolsize=20)
line.render('GDP.data.html')


# # # 柱状图和折线图结合
from pyecharts import Bar, Line,Overlap #导入相关模块
line = Line('') #创建一个实例对象
line.add('国内生产总值',GDP_data.时间,GDP_data['国内生产总值_当季值(亿元)'],mark_point=['average','max','min'],
         mark_point_symbol='diamond',mark_point_textcolor ='blue')
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


# # # # # #补充，不要求学会
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
iris = pd.read_csv('iris.csv')
#  将数据点分成三部分画，在颜色上有区分度
#ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
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





