# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:35:57 2019

@author: liyin
"""

import numpy as np

import pandas as pd
import os

path3=r'C:\data\measure'
df=pd.read_json(os.path.join(path3,'shoulder_records.json'))
df.columns
df['fh'].describe()

df.to_excel(os.path.join(path3,'shoulder_records.xlsx'))

df['dx'].describe()
df['dy'].describe()

df['dw']=np.sqrt(df['dx']**2+df['dy']**2)

df['dw_norm']=df['dw']*df['height']/df['fh']

X=df[['dw_norm']]
X_train=df[['dw_norm']]
Y=df['shoulder']


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y)
Y_pred=model.predict(X_train)
diff=Y_pred-Y

diff.describe()

diff.plot(kind='hist')

df['diff']=diff
df.to_excel(os.path.join(path3,'records_shoulder.xlsx'))

#diff=np.abs(diff)
##dir(diff)
#d=diff.to_numpy()
#d[np.where(d<1)]

import matplotlib.pyplot as plt
plt.figure(figsize=(16,9))
data=df[['dw_norm','shoulder']]
data.plot.scatter(x='dw_norm',y='shoulder')



