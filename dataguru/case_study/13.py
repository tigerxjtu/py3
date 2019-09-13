# -*- coding: utf-8 -*-
"""
Created on Sun Jun 04 11:14:08 2017

@author: tiger
"""

#引入包
from __future__ import division, print_function
from future_builtins import ascii, filter, hex, map, oct, zip

import pandas as pd  
from pandas.tools.rplot import RPlot, TrellisGrid, GeomPoint,\
    ScaleRandomColour  
import numpy as np  
from scipy.stats import uniform 
import statsmodels.api as sm  
import statsmodels.formula.api as smf 
import matplotlib.pyplot as plt  

#读入数据
dodgers = pd.read_csv("d:/data/dodgers.csv")

print(u"\ndodgers data frame ---------------")
#观众人数以千为单位
dodgers['attend_000'] = dodgers['attend']/1000
#查看数据
print(pd.DataFrame.head(dodgers)) 

mondays = dodgers[dodgers['day_of_week'] == 'Monday']
tuesdays = dodgers[dodgers['day_of_week'] == 'Tuesday']
wednesdays = dodgers[dodgers['day_of_week'] == 'Wednesday']
thursdays = dodgers[dodgers['day_of_week'] == 'Thursday']
fridays = dodgers[dodgers['day_of_week'] == 'Friday']
saturdays = dodgers[dodgers['day_of_week'] == 'Saturday']
sundays = dodgers[dodgers['day_of_week'] == 'Sunday']

#以天为单位准备观众数据
data = [mondays['attend_000'], tuesdays['attend_000'], 
    wednesdays['attend_000'], thursdays['attend_000'], 
    fridays['attend_000'], saturdays['attend_000'], 
    sundays['attend_000']]
ordered_day_names = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

# 观测每周每天的数据，以箱线图方式绘制
fig, axis = plt.subplots()
axis.set_xlabel('Day of Week')
axis.set_ylabel('Attendance (thousands)')
day_plot = plt.boxplot(data, sym='o', vert=1, whis=1.5)
plt.setp(day_plot['boxes'], color = 'black')    
plt.setp(day_plot['whiskers'], color = 'black')    
plt.setp(day_plot['fliers'], color = 'black', marker = 'o')
axis.set_xticklabels(ordered_day_names)
plt.show()

#以月为单位准备数据
april = dodgers[dodgers['month'] == 'APR']
may = dodgers[dodgers['month'] == 'MAY']
june = dodgers[dodgers['month'] == 'JUN']
july = dodgers[dodgers['month'] == 'JUL']
august = dodgers[dodgers['month'] == 'AUG']
september = dodgers[dodgers['month'] == 'SEP']
october = dodgers[dodgers['month'] == 'OCT']

data = [april['attend_000'], may['attend_000'], 
    june['attend_000'], july['attend_000'], 
    august['attend_000'], september['attend_000'], 
    october['attend_000']]
ordered_month_names = ['April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct']

#以月为单位观测观众箱线图数据
fig, axis = plt.subplots()
axis.set_xlabel('Month')
axis.set_ylabel('Attendance (thousands)')
day_plot = plt.boxplot(data, sym='o', vert=1, whis=1.5)
plt.setp(day_plot['boxes'], color = 'black')    
plt.setp(day_plot['whiskers'], color = 'black')    
plt.setp(day_plot['fliers'], color = 'black', marker = 'o')
axis.set_xticklabels(ordered_month_names)
plt.show()

#观测白天/黑夜、天气状况、摇头夜促销与否对观众人数的影响
plt.figure()
plot = RPlot(dodgers,  x = 'temp', y = 'attend_000')
plot.add(TrellisGrid(['day_night', 'skies']))
plot.add(GeomPoint(colour = ScaleRandomColour('bobblehead')))
plot.render(plt.gcf())
plt.show()

#按周天映射以便排序
day_to_ordered_day = {'Monday' : '1Monday', 
     'Tuesday' : '2Tuesday', 
     'Wednesday' : '3Wednesday', 
     'Thursday' : '4Thursday', 
     'Friday' : '5Friday',
     'Saturday' : '6Saturday',
     'Sunday' : '7Sunday'}
dodgers['ordered_day_of_week'] = dodgers['day_of_week'].map(day_to_ordered_day) 

# 按月映射便于排序
month_to_ordered_month = {'APR' : '1April', 
     'MAY' : '2May', 
     'JUN' : '3June', 
     'JUL' : '4July', 
     'AUG' : '5Aug',
     'SEP' : '6Sept',
     'OCT' : '7Oct'}
dodgers['ordered_month'] = dodgers['month'].map(month_to_ordered_month) 

# 将数据集分割为训练集和测试集
np.random.seed(1)
dodgers['runiform'] = uniform.rvs(loc = 0, scale = 1, size = len(dodgers))
dodgers_train = dodgers[dodgers['runiform'] >= 0.33]
dodgers_test = dodgers[dodgers['runiform'] < 0.33]

#摇头夜促销模型表达式
my_model = str('attend ~ ordered_month + ordered_day_of_week + bobblehead')

#模型训练
train_model_fit = smf.ols(my_model, data = dodgers_train).fit()
print(train_model_fit.summary())
#训练集中计算模型拟合数据
dodgers_train['predict_attend'] = train_model_fit.fittedvalues
p = dodgers_train[['attend','predict_attend']].plot(subplots = False, style=['b-o','r-*'])
plt.show()             

#测试集中计算模型拟合数据
dodgers_test['predict_attend'] = train_model_fit.predict(dodgers_test)

p = dodgers_test[['attend','predict_attend']].plot(subplots = False, style=['b-o','r-*'])
plt.show()

#对完整数据集进行模型训练拟合
my_model_fit = smf.ols(my_model, data = dodgers).fit()
print(my_model_fit.summary())

print('\nEstimated Effect of Bobblehead Promotion on Attendance: ',\
    round(my_model_fit.params[13],0))
