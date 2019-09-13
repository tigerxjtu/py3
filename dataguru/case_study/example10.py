# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#读入数据
data = pd.read_csv('d:/data/week10.csv')

loc = data[data['RC']!='FIRST'][['locate_X','Locate_Y','WaferID']]

#数据清洗
#删除重复测试值
for i in range(len(loc)):
    data_del = data[data['locate_X']==loc.iloc[i,0]][data['Locate_Y']==loc.iloc[i,1]][data['WaferID']==loc.iloc[i,2]][data['RC']=='FIRST']
    data = data.drop(data_del.index)

#删除重复记录
data = data.drop_duplicates()   
 
#分布探索
test = data['Test_1']
test = test.dropna()

test = test[test<0.001]
test = (test-test.mean())/test.std()
test.hist(bins=100)

#区分正常芯片与异常芯片
test_nor = data[data['Softbin']==1]['Test_1']
test_ab = data[data['Softbin']!=1]['Test_1']
test_ab =test_ab[test_ab<0.001]
test_nor=test_nor[test_nor<0.001]

test_nor.plot(kind='kde')
test_ab.plot(kind='kde')

#区分不同的site
test_site1 = data[data['Softbin']==1][data['SITE']==1]['Test_1']
test_site2 = data[data['Softbin']==1][data['SITE']==2]['Test_1']
test_site3 = data[data['Softbin']==1][data['SITE']==3]['Test_1']
test_site4 = data[data['Softbin']==1][data['SITE']==4]['Test_1']
test_site5 = data[data['Softbin']==1][data['SITE']==5]['Test_1']
test_site6 = data[data['Softbin']==1][data['SITE']==6]['Test_1']
test_site7 = data[data['Softbin']==1][data['SITE']==7]['Test_1']
test_site8 = data[data['Softbin']==1][data['SITE']==8]['Test_1']
test_site1.plot(kind='kde',label='site1')
test_site2.plot(kind='kde',label='site2')
test_site3.plot(kind='kde',label='site3')
test_site4.plot(kind='kde',label='site4')
test_site5.plot(kind='kde',label='site5')
test_site6.plot(kind='kde',label='site6')
test_site7.plot(kind='kde',label='site7')
test_site8.plot(kind='kde',label='site8')

#正态分布检验
from scipy.stats import kstest

kstest(test_site3,'norm',args=(test_site3.mean(),test_site3.std()))

kstest(test,'t',args=(100,0,1))
kstest((test_nor-test_nor.mean())/test_nor.std(),'t',args=(50,0,1))
kstest((test_ab-test_ab.mean())/test_ab.std(),'t',args=(50,0,1))



test_site1 = data[data['SITE']==1]['Test_1']
test_site2 = data[data['SITE']==2]['Test_1']
test_site3 = data[data['SITE']==3]['Test_1']
test_site4 = data[data['SITE']==4]['Test_1']
test_site5 = data[data['SITE']==5]['Test_1']
test_site6 = data[data['SITE']==6]['Test_1']
test_site7 = data[data['SITE']==7]['Test_1']
test_site8 = data[data['SITE']==8]['Test_1']
test_site1.plot(kind='kde')
test_site2.plot(kind='kde')
test_site3.plot(kind='kde')
test_site4.plot(kind='kde')
test_site5.plot(kind='kde')
test_site6.plot(kind='kde')
#test_site7.plot(kind='kde')
test_site8.plot(kind='kde')


#再看一个测试
test = data['Test_2']
test = test.dropna()

test = test[test<0.00001]
test = (test-test.mean())/test.std()
test.hist(bins=100)

test_nor = data[data['Softbin']==1]['Test_2']
test_ab = data[data['Softbin']!=1]['Test_2']
test_ab =test_ab[test_ab<0.00001]

test_nor.plot(kind='kde')
test_ab.plot(kind='kde')


###基于聚类的离群点判别###

#参数初始化
k = 8 #聚类的类别
threshold =2 #离散点阈值
iteration = 500 #聚类最大循环次数


from sklearn.cluster import KMeans
model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration) #分为k类，并发数4
model.fit(pd.DataFrame(test)) #开始聚类

#标准化数据及其类别
r = pd.concat([test, pd.Series(model.labels_, index = test.index)], axis = 1)  #每个样本对应的类别
r.columns = ['Test1',u'聚类类别'] #重命名表头

norm = []
for i in range(k): #逐一处理
  norm_tmp = r['Test1'][r[u'聚类类别'] == i]-model.cluster_centers_[i]
  norm_tmp = norm_tmp.apply(np.linalg.norm) #求出绝对距离
 # norm.append(norm_tmp/norm_tmp.median()) #求相对距离并添加
  norm.append(norm_tmp) #

norm = pd.concat(norm) #合并

discrete_points = norm[norm > norm.mean()+2.5*norm.std()]

res = data.ix[discrete_points.index]['Softbin']
sum(res!=1)
#效果不好
