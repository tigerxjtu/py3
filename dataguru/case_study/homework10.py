# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:08:02 2017

@author: tiger
"""

import numpy as np
import pandas as pd

#读入数据
data = pd.read_csv('d:/data/week10.csv')

#数据清洗
#删除重复测试值
for i in range(len(loc)):
    data_del = data[data['locate_X']==loc.iloc[i,0]][data['Locate_Y']==loc.iloc[i,1]][data['WaferID']==loc.iloc[i,2]][data['RC']=='FIRST']
    data = data.drop(data_del.index)

#删除重复记录
data = data.drop_duplicates()

data.describe()

d=data['Softbin'].value_counts()
d.count()
len(d.index)
d.iloc[0].value()

e=data['SITE'].value_counts()

def distinct_values(X,col):
    return X[col].value_counts().count()


size=e.count()

ds=[]
for i in range(size):
   ds.append(data[data['SITE']==i+1])
    

threshold =2.5 #离散点阈值
iteration = 500 #聚类最大循环次数

cols=['Test_1','Test_2','Test_3','Test_4','Test_5','Test_6','Test_7']

from sklearn.cluster import KMeans

for j in range(size):
    test=ds[j]
    
    k=distinct_values(test,'Softbin')
    model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration)
    test[cols]=test[cols].apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    model.fit(test[cols])
    
    test['class']=model.labels_
    tmp=test[cols]-model.cluster_centers_[test['class']]
    test['distance']=tmp.apply(np.linalg.norm,axis=1) #求出绝对距离
    grouped=test['Softbin'].groupby(test['class'])
    grouped.value_counts()
    
    all=sum(test['Softbin']!=1) 
    correct=0
    candidates=0
    norm = []
    for i in range(k): #逐一处理
      tmp=test[test['class'] == i]
      l=len(tmp)
      if l<5:
          candidates+=l
          correct=correct+sum(test['Softbin'][test['class'] == i]!=1)
          continue
      t=tmp['distance'].mean()+threshold*tmp['distance'].std()
      tmp=tmp[tmp['distance']>t]
      candidates+=len(tmp)
      correct+=sum(tmp['Softbin']!=1)
      
    
#    norm = pd.concat(norm) #合并
#    
#    discrete_points = norm[norm > norm.mean()+threshold*norm.std()]
#    candidates+=len(discrete_points)
#    res = test.ix[discrete_points.index]['Softbin']
#    correct+=sum(res!=1)
    
    print "Site=%d, all=%d, find candidates=%d, and corrects=%d" %(j+1,all,candidates,correct)

#sum(test['Softbin']!=1) 

