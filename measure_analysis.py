# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:35:57 2019

@author: liyin
"""

import numpy as np

data=np.array(range(100))
data2=np.array(range(0,200,2))
index=np.where(data>10)
data=data[index]
data2=data2[index]

index=np.where(data<20)
data=data[index]
data2=data2[index]

data[np.where(data==15)]

data
data2

data.shape

import pandas as pd
import os

path3=r'C:\data\measure'
df=pd.read_json(os.path.join(path3,'records.json'))
df.columns
df['fw'].describe()
df['fh'].describe()

df['bw'].describe()
df['bh'].describe()

df.to_excel(os.path.join(path3,'records.xlsx'))

df['front']=df['fw']*df['height']/df['fh']
df['front'].describe()

df['side']=df['sw']*df['height']/df['sh']
df['side'].describe()

df['back']=df['bw']*df['height']/df['bh']
df['back'].describe()

X=df[['front','side']]
df['front2']=X['front']*X['front']
df['side2']=X['side']*X['side']
df['front_side']=X['front']*X['side']
df['front_back']=(df['front']+df['back'])/2
X_train=df[['front2','side2','front_side']]
Y=df['neck']

#X_train=df[['front','side','back']]
X_train=df[['front','side','front2','side2','front_side']]

X_train=df[['front_back','side']]

#from sklearn.decomposition import PCA
#pca=PCA(n_components=2)
#X_train=pca.fit_transform(X_train)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y)
Y_pred=model.predict(X_train)
diff=Y_pred-Y

diff.describe()

diff.plot(kind='hist')

df['diff']=diff
df.to_excel(os.path.join(path3,'records.xlsx'))

diff=np.abs(diff)
#dir(diff)
d=diff.to_numpy()
d[np.where(d<1)]

import matplotlib.pyplot as plt
plt.figure(figsize=(16,9))
data=df[['front','side','neck','front2','side2','front_side']]
data.plot.scatter(x='front_side',y='neck')



