# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:02:35 2017

@author: tiger
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import os

def get_times(path):
    import pandas as pd
    from datetime import datetime
    from datetime import date 
    user=pd.read_excel(path,sheetname=0,header=1)

    #统计发言时间   
    time1=user[u'/基本信息/item/时间']
    pattern=re.compile(r'\d{4}-\d+-\d+ \d{2}:\d{2}')
    time_1=[]
    for time in time1:
        if type(time)==unicode:
            res=pattern.match(time)
            if res:
                time_1.append(res.group())

    times=[]
    for time in time_1:
    #将发言时间转化为datetime格式
        times.append(datetime.strptime(time,"%Y-%m-%d %H:%M"))  
         
    weekdays=[]
    for time in times:
        weekdays.append(date.isoweekday(time))  
    weekdays_count=pd.Series(weekdays).value_counts()

    hours=[]
    for time in times:
        hours.append(time.strftime('%H'))
    hours_count=pd.Series(hours).value_counts() 
    
    return weekdays_count , hours_count


path=u'D:/data/example05/spider'

#user_name=[name.split('.')[0] for name in i[2]] #获取用户名
user_name=[name.split('.')[0] for name in os.listdir(path)] #获取用户名

hours=pd.DataFrame()
weekdays=pd.DataFrame()
for name in user_name:
    weekday,hour=get_times(path+u'/'+name+u'.xls')
    hours[name]=hour
    weekdays[name]=weekday

hours=hours.sort_index()
weekdays=weekdays.sort_index()

hours=hours.fillna(0)
hours_radio=(hours/hours.sum()).T.fillna(0)
hours_scale=((hours-hours.mean())/hours.std()).T.fillna(0)

weekdays=weekdays.fillna(0)
weekdays_radio=(weekdays/weekdays.sum()).T.fillna(0)
weekdays_scale=((weekdays-weekdays.mean())/weekdays.std()).T.fillna(0)

from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

hours=hours.fillna(0).T
weekdays=weekdays.fillna(0).T
                        
Z = linkage(hours_radio, method = 'ward', metric = 'euclidean') #谱系聚类图
P = dendrogram(Z, 0) #画谱系聚类图
plt.show()

Z = linkage(hours_scale, method = 'ward', metric = 'euclidean') #谱系聚类图
P = dendrogram(Z, 0) #画谱系聚类图
plt.show()

k = 4 #聚类的类别
iteration = 500 #聚类最大循环次数

#构建k-means模型
model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration) #分为k类，并发数4
model.fit(hours_scale) #开始聚类

r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
r2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心

#类中心比较
r2.T.plot(figsize=(10,10))
plt.show()
         
res = pd.concat([hours, pd.Series(model.labels_, index = hours_scale.index)], axis = 1)  #详细输出每个样本对应的类别
res.columns = list(hours.columns) + [u'class'] #重命名表头
res.to_excel('d:/data/example05/result.xls') #保存结果

##用户分析    
#用户发帖时间分析
for i  in range(k):
    x=(res[res['class']==i].mean()).max()
    print u'第',i,u"类用户的平均发帖数的最大值是" ,x
    
active=[1,2,3,2]
clas=pd.DataFrame(active,columns=['active'])

maxs=r2.T.max()
clas['hour']=0
for i  in range(k):
    x=r.columns[r.ix[i]==maxs[i]]
    print u'第',i,u"类用户的最活跃时段是" ,x
    clas.iloc[i,1]=x[0]

weekdays2= pd.concat([weekdays, pd.Series(model.labels_, index = hours_scale.index)], axis = 1) 
weekdays2.columns = list(weekdays.columns) + [u'class']

week=[]
for i in range(k):
    x=weekdays2[weekdays2['class']==i].mean()
    x.plot(kind='bar')
    plt.show()
    print x[x>90]
    week.append(x[x>90])

clas['weekday']=week
    
#inter=pd.read_excel(u'D:/data/example05/interest.xlsx',sheetname=0,header=0,index=0)
#
#int1=[]
#int2=[]
#int3=[]
#for i in inter.columns:
#    x=inter.sort_values(by=i,ascending=False)[i]
#    print x
#    int1.append(x.index[0])
#    int2.append(x.index[1])
#    int3.append(x.index[2])
#    
#clas['int1']=int1
#clas['int2']=int2
#clas['int3']=int3

###分词与词频统计###
#结巴分词
from __future__ import unicode_literals
import sys
sys.path.append("../")

import jieba
import jieba.posseg
import jieba.analyse

print weekdays2.ix[u'用户45']
print clas.ix[1] 
print hours.ix[u'用户45']
   
user_45=pd.read_excel(u'D:/data/example05/spider/用户45.xls',sheetname=0,header=1)
content=user_45[u'/基本信息/item/发言内容']      

content=content[content!=u'/基本信息/item/发言内容'].dropna()


#stopwords = {}.fromkeys([ line.rstrip() for line in open('d:/data/example05/stopwords.txt') ])
hist={} 
for sen in content:          
    for word in jieba.cut(sen,cut_all=False):
#        if word not in stopwords:
            hist[word] = hist.get(word, 0) + 1
hist2=[(key,value) for (key,value) in hist.items() if len(key)>1]
hist_sorted = sorted(hist2, key=lambda d: d[1], reverse=True)
for word,times in hist_sorted[:100] :
    print word,times