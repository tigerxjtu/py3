# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:45:15 2016

@author: tracy
"""

#读入数据
import pandas as pd
data=pd.read_excel(u'd:/data/作业1.xls',index_col=0)

#数据探索分析
import matplotlib.pyplot as plt
data.describe()

pd.value_counts(data[u'销售类型'])
pd.value_counts(data[u'销售模式'])
pd.crosstab(data[u'输出'],data[u'销售类型'])
pd.crosstab(data[u'输出'],data[u'销售模式'])


for col in data.columns:
    if not col in [u'销售类型',u'销售模式',u'输出']:
        fig = plt.figure()
        data[col].hist(bins=20, by = data[u'输出'])
        fig.show()

#数据预处理
data=pd.merge(data,pd.get_dummies(data[u'销售模式']),left_index=True,right_index=True)
data=pd.merge(data,pd.get_dummies(data[u'销售类型']),left_index=True,right_index=True)
data['type']=pd.get_dummies(data[u'输出'])[u'正常']
data = data.iloc[:,2:]
del data[u'输出']

from random import shuffle
data_2=data
data = data.as_matrix() #将表格转换为矩阵
shuffle(data)

p = 0.8 #设置训练数据比例
train = data[:int(len(data)*p),:] #前80%为训练集
test = data[int(len(data)*p):,:] #后20%为测试集


        
#模型构建
def cm_plot(y, yp):
  
  from sklearn.metrics import confusion_matrix

  cm = confusion_matrix(y, yp) 
  
  import matplotlib.pyplot as plt 
  plt.matshow(cm, cmap=plt.cm.Greens) 
  plt.colorbar() 
  
  for x in range(len(cm)): 
    for y in range(len(cm)):
      plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
  
  plt.ylabel('True label') 
  plt.xlabel('Predicted label') 
  return plt
  


#构建CART决策树模型
from sklearn.tree import DecisionTreeClassifier #导入决策树模型

treefile = 'd:/data/example/tree.pkl' #模型输出名字
tree = DecisionTreeClassifier() #建立决策树模型
tree.fit(train[:,:25], train[:,25]) #训练

#保存模型
from sklearn.externals import joblib
joblib.dump(tree, treefile)


cm_plot(train[:,25], tree.predict(train[:,:25])).show() #显示混淆矩阵可视化结果
#注意到Scikit-Learn使用predict方法直接给出预测结果。

from sklearn.metrics import roc_curve #导入ROC曲线函数

fpr, tpr, thresholds = roc_curve(test[:,25], tree.predict_proba(test[:,:25])[:,1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of CART', color = 'green') #作出ROC曲线
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果


#构建神经网络模型
from keras.models import Sequential #导入神经网络初始化函数
from keras.layers.core import Dense, Activation #导入神经网络层函数、激活函数

netfile = 'd:/data/example/net.model' #构建的神经网络模型存储路径

net = Sequential() #建立神经网络
net.add(Dense(30,input_dim=25)) #添加输入层（3节点）到隐藏层（10节点）的连接
net.add(Activation('relu')) #隐藏层使用relu激活函数
net.add(Dense(1,input_dim=30)) #添加隐藏层（10节点）到输出层（1节点）的连接
net.add(Activation('sigmoid')) #输出层使用sigmoid激活函数
net.compile(loss = 'binary_crossentropy', optimizer = 'adam') #编译模型，使用adam方法求解

net.fit(train[:,:25], train[:,25], nb_epoch=10, batch_size=1) #训练模型，循环1000次
net.save_weights(netfile) #保存模型

predict_result = net.predict_classes(train[:,:25]).reshape(len(train)) #预测结果变形
'''这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出预测类别，而且两者的预测结果都是n x 1维数组，而不是通常的 1 x n'''


cm_plot(train[:,25], predict_result).show() #显示混淆矩阵可视化结果

from sklearn.metrics import roc_curve #导入ROC曲线函数

predict_result = net.predict(test[:,:25]).reshape(len(test))
fpr, tpr, thresholds = roc_curve(test[:,25], predict_result, pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of LM') #作出ROC曲线
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果              