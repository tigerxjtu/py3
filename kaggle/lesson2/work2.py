import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt

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

X=df[train_cols]
y=df['affairs'].apply(affair)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()

dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)



dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)

y_test = y_test.values
# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    target = 1 if ans[i]>0.5 else 0
    # print(target)
    # print(y_test[i])
    if target == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

# 显示重要特征
plot_importance(model)
plt.show()




