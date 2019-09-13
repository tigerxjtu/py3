# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:09:41 2019

@author: liyin
"""

from  pyecharts import Pie
# # # # # #绘制柱状图
from pyecharts import Bar

import os
os.chdir('c:\data')

import pandas as pd
import numpy as np
df = pd.read_excel('朝阳医院2018年销售数据.xlsx')
from datetime import datetime
df['购药时间'].dropna(inplace=True)
def to_month(s):
    day=s.split()[0]
    date=datetime.strptime(day,'%Y-%M-%d')
    return date.strftime('%Y-%M')

df['购药时间']=df['购药时间'].apply(to_month)
grouped = df[['应收金额','销售数量']].groupby(df['购药时间'])
data = grouped.sum()

bar =Bar("朝阳医院2018年销售数据",title_pos='left',width=800) #创建一个实例对象
bar.add("销售量",data.index ,data['销售数量'], is_legend_show =True ) #是否显示顶端图例
bar.add("销售金额",data.index ,data['应收金额'],is_legend_show =True ) #是否显示顶端图例
#bar.show_config()
#bar.render('sales.html')

from pyecharts import Line,Overlap
line = Line('朝阳医院2018年销售数据',title_pos='left')
line.add('销售量',data.index ,data['销售数量'],mark_point=['average','max','min'],
         mark_point_symbol='diamond',mark_point_textcolor='blue')
line.add("销售金额",data.index ,data['应收金额'],mark_point=['max'], is_smpooth=True, mark_line=['average','max','min'],
         mark_point_symbol='arrow',mark_point_symbolsize=20)
overlap = Overlap()
overlap.add(line)
overlap.add(bar)
overlap.render('sales.html')



from pyecharts import Scatter3D
exp = pd.read_csv('creditcard_exp.csv')
exp.gender.value_counts()
scatter3d = Scatter3D('avg_exp，Income和high_avg的关系图')
male = pd.DataFrame({'avg_exp':exp.avg_exp[exp.gender==0].values,
                      'Income':exp.Income[exp.gender==0].values,
                      'high_avg':exp.high_avg[exp.gender == 0].values})

male = list(male.values)

female = pd.DataFrame({'avg_exp':exp.avg_exp[exp.gender==1].values,
                      'Income':exp.Income[exp.gender==1].values,
                      'high_avg':exp.high_avg[exp.gender == 1].values})
female= list(female.values)


scatter3d.add("male", male, symbol_size =8,mark_point_symbol='arrow',color='red')
scatter3d.add("female", female, symbol_size =12)
scatter3d.render('exp.html')
