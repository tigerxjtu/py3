import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical

df=pd.read_csv(r"C:\projects\python\data\dataguru\Affairs.csv")

def gender(sex):
    if sex=='male':
        return 1
    return 0

def children(flag):
    if flag=='yes':
        return 1
    return 0

def affair(affairs):
    if affairs>1:
        return 1
    return 0

df['gender']=df['gender'].apply(gender)
df['children']=df['children'].apply(children)

def onehot(df,col,num):
    for i in range(1,num+1):
        new_col = '%s_%d'%(col,i)
        df[new_col]=0
        df[new_col][df[col]==i]=1

onehot(df,'occupation',7)

train_cols = ["gender","age","yearsmarried","children","religiousness","education","rating"]
onehot_cols = ['%s_%d'%('occupation',i) for i in range(1,8)]
train_cols = train_cols + onehot_cols

X=df[train_cols].values
y=to_categorical(df['affairs'].apply(affair).values)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=Sequential()
model.add(Dense(units=14,input_dim=x_train.shape[1],activation='tanh'))
model.add(Dense(units=16,activation='tanh'))
model.add(Dense(units=2,activation='softmax'))
sgd=SGD(lr=0.2)

model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=200)

loss,accuracy=model.evaluate(x_test,y_test)

print('\ntest loss:',loss)
print('accuracy:',accuracy)