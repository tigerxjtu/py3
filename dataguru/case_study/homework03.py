# -*- coding: utf-8 -*-


#加载所需的库
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


#读取数据
file_path=u'd:/data/example03/data2.csv'
data=pd.read_csv(file_path,header=0)

#数据预分析
data.describe() #基本统计量
data.skew() #偏度
data.kurt() #超额峰度,大于0表示比正态分布陡峭，小于0表示比正态分布平缓
data.corr() #相关矩阵

data.iloc[:,:6].hist()

#相关分析
np.round(data.corr(method = 'pearson'), 2)

##模型构建
#Lasso变量选择
from sklearn.linear_model import LassoLars
model = LassoLars(alpha=0.1)
model.fit(data.iloc[:15,0:6],data.iloc[:15,6])
np.round(pd.DataFrame(model.coef_),2) #各个特征的系数

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

data.index = range(1994, 2009)

data.loc[2010] = None
data.loc[2011] = None
l = ['x1', 'x4']
for i in l:
  f = GM11(data[i][range(1994, 2011)].as_matrix())[0]
  data[i][2014] = f(len(data)-1) #2014年预测结果
  data[i][2015] = f(len(data)) #2015年预测结果
  data[i] = data[i].round(2) #保留两位小数

outputfile=u'D:/data/example03/data2_GM11.xls'
data[l+['y']].to_excel(outputfile) #结果输出

#神经网络预测模型
inputfile = 'D:/data/example03/data2_GM11.xls'
outputfile = 'D:/data/example03/revenue2.xls' #神经网络预测后保存的结果
modelfile = 'D:/data/example03/1-net.model' #模型保存路径
data = pd.read_excel(inputfile) #读取数据
feature = ['x1', 'x4'] #特征所在列

data_train = data.loc[range(1994,2010)].copy() #取2010年前的数据建模
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean)/data_std #数据标准化
x_train = data_train[feature].as_matrix() #特征数据
y_train = data_train['y'].as_matrix() #标签数据

from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential() #建立模型
model.add(Dense(input_dim=2, output_dim=6))
model.add(Activation('relu')) #用relu函数作为激活函数，能够大幅提供准确度
model.add(Dense(input_dim=6, output_dim=1))
model.compile(loss='mean_squared_error', optimizer='adam') #编译模型
model.fit(x_train, y_train, nb_epoch = 10000, batch_size = 16) #训练模型，学习一万次
model.save_weights(modelfile) #保存模型参数

#预测，并还原结果。
x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()
data[u'y_pred'] = model.predict(x) * data_std['y'] + data_mean['y']
data.to_excel(outputfile)

import matplotlib.pyplot as plt #画出预测结果图
p = data[['y','y_pred']].plot(subplots = True, style=['b-o','r-*'])
plt.show()
