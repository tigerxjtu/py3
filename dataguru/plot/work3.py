# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:11:17 2019

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# 支持中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

os.chdir(r'D:\BaiduNetdiskDownload\Python数据可视化实战\第三章')

creditcard = pd.read_csv('creditcard_exp.csv')
creditcard.info()
creditcard['edu_class'].value_counts().index
colors_creditcard = ['steelblue','indianred','green','blue']
edu_class =[0,1,2,3]
marker_class =['o','s','x','*']
for i in range(0,4):
    plt.scatter(x=creditcard.Age[creditcard['edu_class'] ==edu_class[i]], 
                y=creditcard.Income[creditcard['edu_class'] == edu_class[i]], s=20,
                color=colors_creditcard[i], marker=marker_class[i], label=edu_class[i])
plt.xlabel('年龄',fontsize =12, labelpad =20)
plt.ylabel('收入',fontsize = 12,  labelpad =20)
plt.title('不同类别用户的年龄和收入的关系图关系图',fontsize =12)
plt.legend(loc='upper left')
plt.show()


Titanic = pd.read_csv('titanic_train.csv')
Titanic.info()
Titanic.head()
Titanic['Pclass'].value_counts()
Titanic['Survived'].value_counts()
# # # 堆叠图
#group_pclass = Titanic.groupby('Pclass')
#avg_price = group_pclass.aggregate({'price_unit':np.mean}).sort_values('price_unit', ascending = False)

temp = pd.crosstab(Titanic['Pclass'],Titanic['Survived'])
plt.bar(x= temp.index.values,height= temp[0],color='steelblue',label='not survived',tick_label = temp.index.values)
plt.bar(x= temp.index.values,height= temp[1],bottom =temp[0], color='green',label='survived',
        tick_label = temp.index.values)

plt.ylabel('人数')
plt.title('游客生还情况')
plt.legend(loc=2, bbox_to_anchor=(1.02,0.8)) #图例显示在外面
plt.show()

# # # 堆叠图占比
temp = pd.crosstab(Titanic['Pclass'],Titanic['Survived'])
temp = temp.div(temp.sum(1).astype(float), axis=0)
plt.bar(x= temp.index.values,height= temp[0],color='steelblue',label='not survived',tick_label = temp.index.values)
plt.bar(x= temp.index.values,height= temp[1],bottom =temp[0], color='green',label='survived',
        tick_label = temp.index.values)
plt.ylabel('比例')
plt.title('游客生还情况占比图')
plt.legend(loc=2, bbox_to_anchor=(1.02,0.8)) #图例显示在外面
plt.show()


# # # 垂直交错条形图
temp = pd.crosstab(Titanic['Pclass'],Titanic['Survived'])
bar_width = 0.2 #设置宽度
pclass = temp.index.values #取出季度名称
np.array(temp[0].values)
plt.bar(x= np.arange(0,3),height=temp[0],color='steelblue',label='not survived',width = bar_width)
plt.bar(x= np.arange(0,3) + bar_width,height= temp[1], color='green',label='survived',width=bar_width)
plt.xticks(np.arange(3)+0.1,pclass,fontsize=12)
plt.ylabel('人数',fontsize=15)
plt.xlabel('等级',fontsize=15)
plt.title('游客生还情况',fontsize=20)
plt.legend(loc = 'upper left')
plt.show()

