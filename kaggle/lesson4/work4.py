import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
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


def modelfit(alg, dtrain, ytrain, dtest,ytest, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values, label=ytrain.values)
        xgtest = xgb.DMatrix(dtest.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    model = alg.fit(dtrain, ytrain, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(ytrain.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(ytrain, dtrain_predprob))

    #     Predict on testing data:
    dtest_predprob = alg.predict_proba(dtest)[:, 1]
    print('AUC Score (Test): %f' % metrics.roc_auc_score(ytest, dtest_predprob))

    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    return model

xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb1, X_train, y_train, X_test, y_test)


#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
param_test1 = {
    'max_depth':[i for i in range(2,5)],
    'min_child_weight':[i for i in range(1,4)]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,min_child_weight=1,
                                         gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                       param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train,y_train)
print(gsearch1.grid_scores_)
print(gsearch1.best_params_, gsearch1.best_score_)

param_test2 = {
    'max_depth':[4,5,6],
    'min_child_weight':[1,2,3]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
                                        min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
                       param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_train,y_train)
print(gsearch2.best_params_, gsearch2.best_score_)

param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
                       param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_train,y_train)
print(gsearch3.best_params_, gsearch3.best_score_)

xgb2 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=1,
        gamma=0.4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

modelfit(xgb2, X_train, y_train, X_test, y_test)


#Grid seach on subsample and max_features
param_test4 = {
    'subsample':[i/10.0 for i in range(6,11)],
    'colsample_bytree':[i/10.0 for i in range(6,11)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
                                        min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
                       param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(X_train,y_train)
print(gsearch4.best_params_, gsearch4.best_score_)


param_test5 = {
    'subsample':[i/100.0 for i in range(75,95,5)],
    'colsample_bytree':[i/100.0 for i in range(85,104,5)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
                                        min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
                       param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(X_train,y_train)
print(gsearch5.best_params_, gsearch5.best_score_)

#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
param_test6 = {
    'reg_alpha':[0,1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
                                        min_child_weight=1, gamma=0.4, subsample=0.9, colsample_bytree=0.95,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
                       param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(X_train,y_train)
print(gsearch6.best_params_, gsearch6.best_score_)

xgb3 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=1,
        gamma=0.4,
        subsample=0.9,
        colsample_bytree=0.95,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

model = modelfit(xgb3, X_train, y_train, X_test, y_test)

# 显示重要特征
plot_importance(model)
plt.show()




