# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 10:05:55 2017

@author: tiger
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import KMeans

df=pd.read_excel(u'd:/data/国内某航空公司会员数据.xls',sheetname=1)
drop_cols=['FFP_DATE','FIRST_FLIGHT_DATE','LOAD_TIME','LAST_FLIGHT_DATE','TRANSACTION_DATE']
df=df.drop(drop_cols,1)
df['age'].hist(bins=10)
pd.value_counts(df['GENDER'])
diff_cols=['MEMBER_NO','GENDER','age','WORK_CITY',
           'WORK_PROVINCE','WORK_COUNTRY','EXCHANGE_COUNT',
           'DAYS_FROM_BEGIN_TO_FIRST','DAYS_FROM_LAST_TO_END']
#df=df.drop(diff_cols,1)
df.describe()
df=df.dropna()
for col in df.columns:
    if not col in diff_cols:
        fig = plt.figure()
        ax=fig.add_subplot(1,1,1)
        df[col].hist(bins=20)
        ax.set_title(col)
        fig.show()
cols=df.columns.difference(diff_cols)  
data_sd = 1.0*(df[cols] - df[cols].mean())/df[cols].std()
#data_sd=df[cols]/df[cols].max()
data_sd.isnull().any()
data_sd=data_sd.dropna(axis=1)


#聚类数目
#会跑死机器
#Z = linkage(data_sd, method = 'ward', metric = 'euclidean') #谱系聚类图
#P = dendrogram(Z, 0) #画谱系聚类图
#plt.show()

#参数初始化
k = 4 #聚类的类别
iteration = 500 #聚类最大循环次数

#构建k-means模型
model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration) #分为k类，并发数4
model.fit(data_sd) #开始聚类

 #简单打印结果
r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
r2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心

r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(data_sd.columns) + [u'class'] #重命名表头
print(r)

#类中心比较
cols=data_sd.columns
r[cols].plot(figsize=(10,10))
plt.show()

#详细输出原始数据及其类别
res = pd.concat([df, pd.Series(model.labels_, index = df.index)], axis = 1)  #详细输出每个样本对应的类别
res.columns = list(df.columns) + [u'class'] #重命名表头

pd.crosstab(pd.qcut(res['age'],q=8),res['class'])
pd.crosstab(res['GENDER'],res['class'])
res.to_excel('d:/data/result.xls') #保存结果

res[[u'age',u'class']].hist(by='class')
res[u'age'].groupby(res['class']).mean()

def density_plot(data): #自定义作图函数
  import matplotlib.pyplot as plt
  plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
  p = data.plot(kind='kde', linewidth = 2, subplots = True, sharex = False,figsize=(10,80) )
  [p[i].set_ylabel(u'密度',fontproperties='SimHei') for i in range(k)]
  plt.legend()
  return plt

pic_output = 'd:/data/airline_' #概率密度图文件名前缀
df.drop('ELITE_POINTS_SUM_YR_1',axis=1)
for i in range(k):
  density_plot(df[res[u'class']==i]).savefig(u'%s%s.png' %(pic_output, i))       
