# 这是第十章代码
#导入相关库
import  pandas as pd
import numpy as np
import os
os.chdir(r'D:\炼数成金数据可视化\第十章\数据')
#==============================================================================
# 10.1 Plotly介绍
#==============================================================================
# # # # # # 10.1 Plotly介绍
# # py.plot 是绘图的主要函数
# link_text 右下角显示的文字，默认为Export to plotly.ly
# validate为True，确保所有关键字有效的
# filename是保存的路劲
import plotly as py
from plotly.graph_objs import Scatter
trace0 = Scatter(x = [1,2,3,4],y = [10,15,13,17])
trace1 = Scatter(x = [1,2,3,4],y = [6,5,11,9])
data = [trace0,trace1]
py.offline.plot(data,filename ='fth.html')
# 代码解释
# 首先定义一个trace0的变量，用于保存绘图数据
# 每个绘图对象都由Plotly模块里面graph_objs图像对象中的子模块来定义
#==============================================================================
# 10.2 Plotly常见图形
#==============================================================================
# # # # # # 散点图
# 读取数据
# mode--定义图形类型，散点或者线形图
pyplot  = py.offline.plot
GDP_data = pd.read_excel('国民经济核算季度数据.xlsx')
trace1 = Scatter(x =GDP_data.loc[:,'时间'], y = GDP_data.loc[:,'第一产业增加值_当季值(亿元)'],
                 mode ='markers',name = '第一产业增加值_当季值')
trace2 = Scatter(x =GDP_data.loc[:,'时间'], y =GDP_data.loc[:,'第二产业增加值_当季值(亿元)'],
                 mode ='lines',name = '第二产业增加值_当季值')
trace3 = Scatter(x =GDP_data.loc[:,'时间'], y =GDP_data.loc[:,'第三产业增加值_当季值(亿元)'],
                 mode ='markers+lines',name = '第三产业增加值_当季值')
# mode--定义图形类型，散点或者线形图
data  = [trace1,trace2,trace3]
pyplot(data,filename ='fth.html')

# # #样式设置
# mode--定义图形类型，散点或者线形图
# marker和line分别定义 点的属性和线条
import plotly.graph_objs as pygo
trace1 = Scatter(x =GDP_data.loc[:,'时间'], y =GDP_data.loc[:,'第一产业增加值_当季值(亿元)'],
                 mode ='markers',name = '第一产业增加值_当季值',marker=dict(size=10,color='steelblue'))
trace2 = Scatter(x =GDP_data.loc[:,'时间'], y =GDP_data.loc[:,'第二产业增加值_当季值(亿元)'],
                 mode ='lines',name = '第二产业增加值_当季值',line = dict(width =2,color ='red'))
trace3 = Scatter(x =GDP_data.loc[:,'时间'], y =GDP_data.loc[:,'第三产业增加值_当季值(亿元)'],
                 mode ='markers+lines',name = '第三产业增加值_当季值',
                 marker=dict(size=10,color='green'),line = dict(width =2,color ='steelblue'))
layout = dict(title= '中国三产业变化趋势图',yaxis =dict(zeroline=True), #显示y轴的0刻度线
                  xaxis =dict(zeroline=False,tickangle =-30)) #不显示x轴的0刻度线
# mode--定义图形类型，散点或者线形图
# marker和line分别定义 点的属性和线条
data  = [trace1,trace2,trace3]
fig = dict(data = data, layout =layout)
pyplot(fig,filename ='fth.html')

# # # # # # 线性图
pyplot  = py.offline.plot
import  plotly.graph_objs as go
jd_stock  = pd.read_excel('data.xlsx', sep =',',header=None,names =['name','date','opening_price','closing_price',
                                                                 'lowest_price','highest_price','volume'])
trace1 = go.Scatter(x =jd_stock.date ,y = jd_stock.opening_price,name = '开盘价',mode ='lines',
                line = dict(width = 2,color='red'))
trace2 = go.Scatter(x =jd_stock.date ,y = jd_stock.closing_price,name = '收盘价',mode ='lines+markers',
                line = dict(width = 2,color='blue'))
trace3 = go.Scatter(x =jd_stock.date ,y = jd_stock.lowest_price,name = '最低价',mode ='lines',
                line = dict(width = 2,color='darkblue'))
trace4 = go.Scatter(x =jd_stock.date ,y = jd_stock['highest_price'],name = '最高价',mode ='markers',
                marker = dict(size = 3,color='darkblue'))

data = [trace1, trace2, trace3, trace4]
#go.Layout就可以创建图层对象
layout = go.Layout(title = '股票趋势图',xaxis = dict(title = '日期'),yaxis = dict(title = '价格'),
                   legend=dict(x=1,y =0.5,font = dict(size=16,color='black')))
fig = go.Figure(data=data,layout = layout)# data与layout组成一个图象对象
pyplot(fig, filename='styled_line') #输出

# # # # # # 柱状图
import  plotly.graph_objs as go
pyplot  = py.offline.plot
Titanic = pd.read_csv('titanic_train.csv')
P_class = Titanic.Pclass.value_counts()#统计P的等级
trace_basic = [go.Bar(x =P_class.index.tolist(),y = P_class.values.tolist(),
                      marker=dict(color=["red", "blue","green"]))]
layout = go.Layout(title = '柱状图', xaxis =dict(title ='仓位等级'))
figure_basic = go.Figure(data =trace_basic,layout=layout)# data与layout组成一个图象对象
pyplot(figure_basic, filename='styled_line.html') #输出

# # # # # # 柱状簇
Industry_GDP = pd.read_excel('Industry_GDP.xlsx')
G1 = Industry_GDP[Industry_GDP['Industry_Type'] =='第一产业']
G2 = Industry_GDP[Industry_GDP['Industry_Type'] =='第二产业']
G3 = Industry_GDP[Industry_GDP['Industry_Type'] =='第三产业']

trace_1 = go.Bar(x =G1.Quarter, y = G1.GDP,name ='第一产业')
trace_2 = go.Bar(x =G2.Quarter, y = G2.GDP,name ='第二产业')
trace_3 = go.Bar(x =G3.Quarter, y = G3.GDP,name ='第三产业')

trace = [trace_1,trace_2,trace_3]
layout = go.Layout(title = '三大产业的GDP', xaxis =dict(title ='季度'))
# figure
figure = go.Figure(data = trace,layout=layout)
pyplot(figure,filename='styled_line.html') #输出

# # #层叠柱状图
Industry_GDP = pd.read_excel('Industry_GDP.xlsx')
G1 = Industry_GDP[Industry_GDP['Industry_Type'] =='第一产业']
G2 = Industry_GDP[Industry_GDP['Industry_Type'] =='第二产业']
G3 = Industry_GDP[Industry_GDP['Industry_Type'] =='第三产业']

trace_1 = go.Bar(x =G1.Quarter, y = G1.GDP,name ='第一产业')
trace_2 = go.Bar(x =G2.Quarter, y = G2.GDP,name ='第二产业')
trace_3 = go.Bar(x =G3.Quarter, y = G3.GDP,name ='第三产业')

trace = [trace_1,trace_2,trace_3]
layout = go.Layout(title = '三大产业的GDP', xaxis =dict(title ='季度'),barmode ='stack')
# figure
figure = go.Figure(data = trace,layout=layout)
pyplot(figure,filename='styled_line.html') #输出


# # # 堆叠图占比
temp = pd.crosstab(Industry_GDP['Quarter'],Industry_GDP['Industry_Type'],values=Industry_GDP['GDP'],aggfunc=np.sum,
                   normalize='index')
trace_1 = go.Bar(x =temp.index.values, y = temp.第一产业.values,name ='第一产业')
trace_2 = go.Bar(x =temp.index.values, y = temp.第二产业.values,name ='第二产业')
trace_3 = go.Bar(x =temp.index.values, y = temp.第三产业.values,name ='第三产业')

trace = [trace_1,trace_2,trace_3]
layout = go.Layout(title = '三大产业的GDP', xaxis =dict(title ='季度'),barmode ='stack')
# figure
figure = go.Figure(data = trace,layout=layout)
pyplot(figure,filename='styled_line.html') #输出


# # #水平柱状图
import  plotly.graph_objs as go
pyplot  = py.offline.plot
Titanic = pd.read_csv('titanic_train.csv')
P_class = Titanic.Pclass.value_counts()#统计P的等级
trace_basic = [go.Bar(x =P_class.values.tolist(),y = P_class.index.tolist(),
                      marker=dict(color=["red", "blue","green"]), orientation='h')]
layout = go.Layout(title = '柱状图', yaxis =dict(title ='仓位等级'))
figure_basic = go.Figure(data =trace_basic,layout=layout)# data与layout组成一个图象对象
pyplot(figure_basic, filename='styled_line.html') #输出

# # #层叠柱状图
Industry_GDP = pd.read_excel('Industry_GDP.xlsx')
G1 = Industry_GDP[Industry_GDP['Industry_Type'] =='第一产业']
G2 = Industry_GDP[Industry_GDP['Industry_Type'] =='第二产业']
G3 = Industry_GDP[Industry_GDP['Industry_Type'] =='第三产业']

trace_1 = go.Bar(y =G1.Quarter, x = G1.GDP,name ='第一产业',orientation='h')
trace_2 = go.Bar(y =G2.Quarter, x = G2.GDP,name ='第二产业',orientation='h')
trace_3 = go.Bar(y =G3.Quarter, x = G3.GDP,name ='第三产业',orientation='h')

trace = [trace_1,trace_2,trace_3]
layout = go.Layout(title = '三大产业的GDP', yaxis =dict(title ='季度',titlefont =dict(size=22,color = 'red')), #大小和颜色
                   barmode ='stack')
# figure
figure = go.Figure(data = trace,layout=layout)
pyplot(figure,filename='styled_line.html') #输出

# # # # # # 直方图
Titanic = pd.read_csv('titanic_train.csv')
data = [go.Histogram(x = Titanic['Age'], histnorm ='probability',marker = dict(color = 'blue'))]
pyplot(data,filename='styled_line.html') #输出

# # #重叠直方图
trace_1 = go.Histogram(x = Titanic.loc[Titanic['Sex'] =='male','Age'], histnorm ='probability',\
                       name='男性年龄分布图' )
trace_2 = go.Histogram(x = Titanic.loc[Titanic['Sex'] =='female','Age'], histnorm ='probability',\
                       name='女性年龄分布图' )
trace  = [trace_1,trace_2]
layout = go.Layout(title = '男女年龄分布图',barmode='overlay')
figure = go.Figure(data = trace,layout=layout) # data与layout组成一个图象对象
pyplot(figure,filename='styled_line.html') #输出

###将直方图和核密度图融合在一起
import plotly.figure_factory as ff
trace_1 = Titanic.loc[(Titanic['Sex'] =='male')&(Titanic['Age'].notnull()),'Age']
trace_2 = Titanic.loc[(Titanic['Sex'] =='female')&(Titanic['Age'].notnull()),'Age']
trace  = [trace_1,trace_2]
group_labels =  [ '男性','女性']
fig = ff.create_distplot(trace,group_labels,bin_size=2)
pyplot(fig,filename='styled_line.html') #输出
#
# # # # # #饼图
labels = ['股票','债券','现金','衍生品','其他']
values = [33.7,20.33,9.9,8.6,27.47]
trace = [go.Pie(labels = labels, values = values)]
layout = go.Layout(title = '基金配置比例图')
fig = go.Figure(data = trace,layout=layout) # data与layout组成一个图象对象
pyplot(fig,filename='styled_line.html') #输出

# 以Titanic号数据
Pclass = Titanic.Pclass.value_counts()
trace = [go.Pie(labels = ['高级','低级','中级'], values = Pclass.values)]
layout = go.Layout(title = '仓位等级分布情况')
fig = go.Figure(data = trace,layout=layout) # data与layout组成一个图象对象
pyplot(fig,filename='styled_line.html') #输出


# # # # # #三维图
creditcard = pd.read_csv('creditcard_exp.csv')
import plotly.graph_objs as go
# 第一部分数据，男性收入，年龄和消费支出
trace1 = go.Scatter3d(
    x=creditcard['avg_exp'][creditcard['gender'] ==1],
    y=creditcard['Age'][creditcard['gender'] ==1],
    z=creditcard['Income'][creditcard['gender'] ==1],
    mode='markers',
    name = '男性',
    # 设定点的大小颜色透明度
    marker=dict(
        size=7,
        # 设定点的轮廓颜色和宽度
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

# 第二部分数据，女性收入，年龄和消费支出
trace2 = go.Scatter3d(
    x=creditcard['avg_exp'][creditcard['gender'] ==0],
    y=creditcard['Age'][creditcard['gender'] ==0],
    z=creditcard['Income'][creditcard['gender'] ==0],
    mode='markers',
    name ='女性',
    marker=dict(
        color='red',
        size=7,
        symbol='circle',
        line=dict(
            color='red',
            width=1
        ),
        opacity=0.9,
    )
)
data = [trace1, trace2]
layout = go.Layout(
    title='不同性别三个变量的关系图')
fig= go.Figure(data=data, layout=layout)
pyplot(fig,filename='styled_line.html') #输出

