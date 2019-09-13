# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:33:23 2019

@author: liyin
"""

import  pandas as pd
import numpy as np
import os
os.chdir(r'c:\data')

import plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import Scatter

taobao=pd.read_csv('taobao_data.csv')

taobao_grouped=taobao['成交量'].groupby(taobao['位置'])
taobao_sum=taobao_grouped.sum()

x=taobao_sum.index.tolist()
y=taobao_sum.values.tolist()
pyplot  = py.offline.plot
data = [go.Bar(x =x,y = y)]
layout = go.Layout(title = '柱状图', yaxis =dict(title ='成交量'))
taobao_fig = go.Figure(data =data,layout=layout)# data与layout组成一个图象对象
pyplot(taobao_fig, filename='taobao.html') #输出

pyplot  = py.offline.plot
data=[Scatter(y =taobao['价格'].values.tolist(), x = taobao['成交量'].values.tolist(),
                 mode ='markers',name = '成交量和价格')]
# mode--定义图形类型，散点或者线形图
layout = go.Layout(title = '淘宝成交量和价格关系图', xaxis =dict(title ='成交量'),yaxis =dict(title ='价格'))
taobao_fig = go.Figure(data =data,layout=layout)# data与layout组成一个图象对象
pyplot(taobao_fig, filename='taobao.html') #输出


film=pd.read_excel(r'film.xlsx')
col='country'
grouped=film['people'].groupby(film[col])
data=grouped.sum()
data.sort_values(ascending=False,inplace=True)
data=data.iloc[:10]
bar = [go.Bar(x =data.index.tolist(),y = data.values.tolist())]
layout = go.Layout(title = '那个国家的电影最流行', yaxis =dict(title ='people'))
taobao_fig = go.Figure(data =bar,layout=layout)# data与layout组成一个图象对象
pyplot(taobao_fig, filename='3-1.html') #输出

col='type'
grouped=film['people'].groupby(film[col])
data=grouped.sum()
data.sort_values(ascending=False,inplace=True)
data=data.iloc[:10]
bar = [go.Bar(x =data.index.tolist(),y = data.values.tolist())]
layout = go.Layout(title = '那个类型的电影最受欢迎', yaxis =dict(title ='people'))
taobao_fig = go.Figure(data =bar,layout=layout)# data与layout组成一个图象对象
pyplot(taobao_fig, filename='3-2.html') #输出


import re
def pingjia(src):
    regexp=re.compile('(\d+).*')
    match=regexp.match(src)
    if match:
        return (int)(match.group(1))
    return 0
film['pj']=film['pingjia'].apply(pingjia)
film1=film.sort_values('pj',ascending=False)
data=film1.iloc[:10][['name','pj']]

bar = [go.Bar(x =data['name'].values.tolist(),y = data['pj'].values.tolist())]
layout = go.Layout(title = '评价人数最多的前10部电影', yaxis =dict(title ='评价数'))
taobao_fig = go.Figure(data =bar,layout=layout)# data与layout组成一个图象对象
pyplot(taobao_fig, filename='3-3.html') #输出

film1=film.sort_values('score',ascending=False)
data=film1.iloc[:10][['name','score']]

bar = [go.Bar(x =data['name'].values.tolist(),y = data['score'].values.tolist())]
layout = go.Layout(title = '评分最高的10部电影', yaxis =dict(title ='score'))
taobao_fig = go.Figure(data =bar,layout=layout)# data与layout组成一个图象对象
pyplot(taobao_fig, filename='3-4.html') #输出

  

house=pd.read_excel('house_info.xlsx')
# # # # # #饼图
data=house['others'].value_counts()
labels = data.index.tolist()
values = data.values.tolist()
trace = [go.Pie(labels = labels, values = values)]
layout = go.Layout(title = '精装毛坯比例图')
fig = go.Figure(data = trace,layout=layout) # data与layout组成一个图象对象
pyplot(fig,filename='4-1.html') #输出

data=house['location'].value_counts()
labels = data.index.tolist()
values = data.values.tolist()
trace = [go.Pie(labels = labels, values = values)]
layout = go.Layout(title = '楼盘朝向比例图')
fig = go.Figure(data = trace,layout=layout) # data与layout组成一个图象对象
pyplot(fig,filename='4-2.html') #输出

data=house['house_type'].value_counts()
labels = data.index.tolist()
values = data.values.tolist()
trace = [go.Pie(labels = labels, values = values)]
layout = go.Layout(title = 'house_type比例图')
fig = go.Figure(data = trace,layout=layout) # data与layout组成一个图象对象
pyplot(fig,filename='4-3.html') #输出

import plotly.figure_factory as ff
pyplot  = py.offline.plot
trace_1 = go.Histogram(x = house['单价'], \
                       name='单价分布图' )
trace_2 = go.Histogram(x = house['total_price'],\
                       name='总价分布图' )
#trace_1=house['单价']
#trace_2=house['total_price']
trace  = [trace_1]
#group_labels =  [ '单价','总价']
#fig = ff.create_distplot(trace,group_labels,bin_size=2)
layout = go.Layout(title = '单价分布图')
figure = go.Figure(data = trace,layout=layout) # data与layout组成一个图象对象
pyplot(figure,filename='4-4.html') #输出

trace  = [trace_2]
#group_labels =  [ '单价','总价']
#fig = ff.create_distplot(trace,group_labels,bin_size=2)
layout = go.Layout(title = '总价分布图')
figure = go.Figure(data = trace,layout=layout) # data与layout组成一个图象对象
pyplot(figure,filename='4-4-2.html') #输出