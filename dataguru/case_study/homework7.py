# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:07:40 2017

@author: tiger
"""
from scipy.stats import chisquare 
import numpy as np
import pandas as pd
import re

from matplotlib import pyplot as plt
%matplotlib inline
from IPython.core.pylabtools import figsize
figsize(11, 9)

file='d:/data/bh2.txt'
records=[]
with open(file) as f:
    for line in f:
        arr = re.split('\s+',line)
        records.extend([a for a in arr if len(a)==9])
print records
ds=[[l for l in i] for i in records ]
data=np.array(ds)
print data.shape
print data[:,0]

def randomtest(data):
    N=len(data)
    data=pd.Series(data)
    if(N<30):
        R=np.nan
        return R
    else:
        p = [0.1*N]*10
        ni=np.zeros([10])
        for i in range(10):
            ni[i] = len(data[data==i])
        p_value = chisquare(ni, p)[1]  
        if(p_value>=0.05):
            R = 0
            return R
        else:
            R=1 
            return R

for i in range(data.shape[1]):
    print "col=%d, r=%d" %(i,randomtest(data[:,i]))
#这里明显感觉随机性检验结果不太对

df=pd.DataFrame(data)
print df.head()

for col in df.columns:
    val=df[col].value_counts()
    print val
    plt.show()
    val.plot(kind='bar')
    
df['f0_2']=df[0]+df[1]+df[2]
df['f0_2'].value_counts().plot(kind='bar')


from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
inputs=range(9)
x_train=df[inputs]
y_train=df['f0_2']
clf.fit(x_train,y_train)

x_test=df.iloc[500][0:9]
print x_test
clf.predict(x_test)


