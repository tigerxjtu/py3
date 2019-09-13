# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from datetime import datetime 
from sklearn.decomposition import PCA,FactorAnalysis
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering,KMeans
from scipy import ndimage
from sklearn import manifold, datasets
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage,dendrogram

#数据读取
inputfile = u'D:/data/example03/data2.csv' #输入的数据文件
data = pd.read_csv(inputfile) #读取数据

#灰色预测模型
def GM11(x0): #自定义灰色预测函数
  import numpy as np
  x1 = x0.cumsum() #1-AGO序列
  z1 = (x1[:len(x1)-1] + x1[1:])/2.0 #紧邻均值（MEAN）生成序列
  z1 = z1.reshape((len(z1),1))
  B = np.append(-z1, np.ones_like(z1), axis = 1)
  Yn = x0[1:].reshape((len(x0)-1, 1))
  [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) #计算参数
  f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) #还原值
  delta = np.abs(x0 - np.array([f(i) for i in range(1,len(x0)+1)]))
  C = delta.std()/x0.std()
  P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)
  return f, a, b, x0[0], C, P #返回灰色预测函数、a、b、首项、方差比、小残差概率

#插值
f = GM11(data['x6'].as_matrix())[0]
for i in reversed(range(5)):
    data['x6'][i]=f(i)

data.corr() #相关矩阵         

data_train = data.copy() #取2014年前的数据建模
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean)/data_std #数据标准化
         
from sklearn.linear_model import LassoLars
model = LassoLars(alpha=0.1)
model.fit(data_train.iloc[:,0:6],data_train['y'])
np.round(pd.DataFrame(model.coef_),4) #各个特征的系数

feature = ['x2', 'x3'] #特征所在列
rows=len(data)
data.loc[rows]=None
data.loc[rows+1]=None

for i in feature:
  f = GM11(data[i][:rows].as_matrix())[0]
  data[i][rows] = f(rows) #2014年预测结果
  data[i][rows+1] = f(rows+1) #2015年预测结果

            
#pca = PCA()
#pca.fit(data)
#pca.explained_variance_ratio_


x_train = data_train[feature][:rows].as_matrix() #特征数据
y_train = data_train['y'][:rows].as_matrix() #标签数据

modelfile = 'D:/data/example03/1-net.model' #模型保存路径
from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential() #建立模型
model.add(Dense(input_dim=2, output_dim=4))
model.add(Activation('relu')) #用relu函数作为激活函数，能够大幅提供准确度
model.add(Dense(input_dim=4, output_dim=1))
model.compile(loss='mean_squared_error', optimizer='adam') #编译模型
model.fit(x_train, y_train, nb_epoch = 5000, batch_size = 16) #训练模型，学习一万次
model.save_weights(modelfile) #保存模型参数

x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()
data[u'y_pred'] = model.predict(x) * data_std['y'] + data_mean['y']
#data.to_excel(outputfile)

import matplotlib.pyplot as plt #画出预测结果图
p = data[['y','y_pred']].plot(subplots = False, style=['b-o','r-*'])
plt.show()