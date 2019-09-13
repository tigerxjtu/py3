# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir(r'D:\data')
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


data  = pd.read_csv('titanic_train.csv')
data['Embarked'].dropna(inplace=True)
df=data['Embarked'].value_counts()
max_index=df.argmax()
indexes=list(df.index)
x=indexes.index(max_index)
max_value=df[max_index]
plt.ylim([0,max_value+100])
df.plot(kind = 'bar', align='center', color = 'steelblue', edgecolor = 'black')
plt.xlabel('类别')
plt.ylabel('数量')
plt.text(x,max_value+50,'maximum:%d' % max_value, color='red')
#plt.annotate('maximum:'+max_value,xy=(0,max_value), xytext=(0,max_value+100))
# 添加标题
plt.title('Embarked分布')
# 显示图形
plt.show()

df=data.Fare
data.Fare.plot(kind = 'hist', bins = 10, color = 'steelblue', edgecolor = 'black', label = '直方图',density = True)
# 绘制核密度图
data.Fare.plot(kind = 'density', color = 'red',label = '核密度图',xlim=[0,data.Fare.max()+5])
# 添加x轴和y轴标签
plt.xlabel('Fare')
plt.ylabel('核密度值')
# 添加标题
plt.title('Fare分布')
# 显示图例
plt.legend()
# 显示图形
plt.show()

Titanic=data
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