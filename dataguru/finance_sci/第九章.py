"""
这是第九章知识点
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import os
os.chdir(r'D:\BaiduNetdiskDownload\Python金融业数据化运营实战\第九章（加密）')
#==============================================================================
# 3.4.1 评分卡建模
#==============================================================================
# 1. 数据获取
df  = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
df.columns



# 2. 数据探索分析
# 查看数据
df.head(10)
df['Loan_Status'].value_counts()
# 相关变量分布
df['ApplicantIncome'].hist(bins=50)
df['LoanAmount'].hist(bins=50)
df.boxplot(column='LoanAmount')
# 离散变量探索
temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],
                       aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

# 通过图形展示离散型变量与因变量之间关系
temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

temp3.div(temp3.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Stacked Bar Chart of Credit_History vs Loan_Status')
plt.xlabel('Credit_History ')
plt.ylabel('Proportion of Loan_Status')

# 查看gender变量与因变量关系
temp3 = pd.crosstab(df['Gender'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
temp3.div(temp3.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Stacked Bar Chart of Gender vs Loan_Status')
plt.xlabel('Gender')
plt.ylabel('Proportion of Loan_Status')

# 3.数据预处理
# 缺失值的统计
df.apply(lambda x: sum(x.isnull()), axis=0)

# 缺失值处理
# 以下是两种填补方法
# 对于连续型变量，用中位数填补
#df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
Loan_Amount_Term_mean=df['Loan_Amount_Term'].median()
df['Loan_Amount_Term'].fillna(Loan_Amount_Term_mean, inplace=True)
# 用分组中位数填补
# 计算分组中位数
LoanAmount_median =df.groupby('Education').LoanAmount.median()
def fage(x):
    return LoanAmount_median.loc[x['Education']]
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

# Replace missing values
# 分组填补中位数，使用两个分类变量
# table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# # Define function to return value of this pivot_table
# def fage(x):
#     return table.loc[x['Self_Employed'],x['Education']]
# # Replace missing values
# df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


# 对于离散型变量用众数填补
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No', inplace=True)
df['Gender'].value_counts()
df['Married'].fillna('Yes', inplace=True)
df['Gender'].fillna('Male', inplace=True)
df['Dependents'].fillna(0, inplace=True)
df['Credit_History'].fillna(1, inplace=True)

# 处理异常值
# 一般先用分布图识别异常值
df['LoanAmount'].hist(bins=30)
df['Loan_Amount_Term'].hist(bins=30)
# 可以使用自然对数
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

# 在进行建模前，需要将离散型变量全部进行哑变量处理
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
for var in var_mod:
    cat_list = 'var' + '_' + var
    cat_list = pd.get_dummies(df[var], prefix=var)
    df1 = df.join(cat_list)
    df = df1

# 原来的离散型变量不参与后续建模分析
df_vars=df.columns.values.tolist()
to_keep=[i for i in df_vars if i not in var_mod]
df_final=df[to_keep]
df_final.columns.values

# 对因变量进行处理
df_final['Loan_Status_new'] = df_final['Loan_Status'].map({'Y':1,'N':0})
# 删除原来的因变量
del df_final['Loan_Status']

# 由于哑变量对每个离散型变量，产生k-1个哑变量
var_delete = [ 'Gender_Male','Married_Yes','Dependents_0','Education_Not Graduate',
               'Self_Employed_Yes',    'Property_Area_Rural' ]
for i in var_delete:
    del df_final[i]
# 查看数据的变量
df_final.columns.values

# 在建模前，需要将数据集拆分为训练和测试集
# 首先确定自变量和因变量
final_vars = df_final.columns.values.tolist()
var_delete = ['Loan_Status_new','Loan_ID']
X=[i for i in final_vars if i not in var_delete ]
df_final_X  = df_final[X]
df_final_y = df_final['Loan_Status_new']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_final_X , df_final_y, test_size=0.3,random_state=0)
# 查看是否有多重共线性
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
# 选择变量
vif =pd.DataFrame(columns=['feature','vif']) #VIF数据集
vif['feature'] = X_train.columns #重新命名
vif['vif'] = [VIF(X_train.values,i) for i in range(X_train.shape[1])] #计算VIF的值
print(vif)

# 删除VIF过高的变量
del  X_train['Loan_Amount_Term']
del X_train['LoanAmount_log']
del  X_test['Loan_Amount_Term']
del X_test['LoanAmount_log']
# 4. 模型建立
# 使用sklearn中的逻辑回归库
from sklearn import linear_model
from sklearn import metrics
clf = linear_model.LogisticRegression()
clf.fit(X_train, Y_train)

# 5. 模型评估
# 对于分类模型可以先看准确率
# 训练集
predict_train = clf.predict(X_train)
accuracy_train = metrics.accuracy_score(predict_train,Y_train)
# 测试集
predict_test = clf.predict(X_test)
accuracy_test = metrics.accuracy_score(predict_test,Y_test)
print ('训练集的准确率: %.4f' %  accuracy_train)
print ('测试集的准确率: %.4f' %  accuracy_test)

# 对于分类模型，一般常用ROC指标来进行评估
# 一般是在测试集上进行评估
# 一般要比较训练集和测试上的效果
prob_train = clf.predict_proba(X_train)[:,1]
fpr_train, sensitivity_train, _ = metrics.roc_curve(Y_train, prob_train)
auc_train = metrics.auc(fpr_train,sensitivity_train)

prob_test = clf.predict_proba(X_test)[:,1]
fpr_test, sensitivity_test, _ = metrics.roc_curve(Y_test, prob_test)
auc_test = metrics.auc(fpr_test,sensitivity_test)
print ('训练集的ROC: %.4f' %  auc_train )
print ('测试集的ROC: %.4f' %  auc_test)
# 测试集上ROC大于0.75，说明模型效果还不错
# 作图
# 可视化结果
import seaborn as sns
sns.set(style='darkgrid',context='notebook',font_scale=1.5) # 设置背景
plt.plot(fpr_train, sensitivity_train,color='black',linestyle="--",label= 'train_set LR (auc =%.3f) ' % (auc_train))
plt.plot(fpr_test,  sensitivity_test,color='blue', linestyle=":", label= 'Test set RF (auc =%.3f) ' % (auc_test))
plt.legend(loc= 'lower right',fontsize= 14)
plt.plot([0,1],[0,1],linestyle='--', color = 'gray',linewidth =2)
plt.xlim([-0.0,1.0])
plt.ylim([-0.0,1.0])
plt.grid()
plt.xlabel("False Positive Rate",fontsize =14)
plt.ylabel('True Positive Rate',fontsize =14)
plt.title("ROC area for train and test",fontsize=14)
plt.show()

### Calculate the KS and AR for the socrecard model
def KS_AR(df, score, target):
    '''
    :param df: the dataset containing probability and bad indicator
    :param score:
    :param target:
    :return:
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all.index = range(len(all))
    all = all.sort_values(by=score,ascending=True)
    #all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    all['totalPcnt'] = all['total'] / all['total'].sum()
    arList = [0.5 * all.loc[0, 'badCumRate'] * all.loc[0, 'totalPcnt']]
    for j in range(1, len(all)):
        ar0 = 0.5 * sum(all.loc[j - 1:j, 'badCumRate']) * all.loc[j, 'totalPcnt']
        arList.append(ar0)
    arIndex = (2 * sum(arList) - 1) / (all['good'].sum() * 1.0 / all['total'].sum())
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return {'AR':abs(arIndex), 'KS': max(abs(KS))}

# 合并数据集，在测试集里面添加预测变量
df_test = pd.DataFrame({'target': Y_test,'score': prob_test })
### 计算KS值和Gini系数
AR_KS = KS_AR(df_test, 'score', 'target')
print('测试集KS和GINI系数分别为: %.3f,%.3f' %(AR_KS['KS'],AR_KS['AR']))
# 以下函数是用来画KS图的
def KS(df, score, target, plot = True):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    :return: KS值
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all['score'] = all.index
    all.index = range(len(all))
    all = all.sort_values(by='score',ascending=True)
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    KS_list = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    KS = max(abs(KS_list))
    if plot:
        # 支持中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(all[score], all['badCumRate'],label='违约客户累计概率')
        plt.plot(all[score], all['goodCumRate'],label='好客户累计概率')
        plt.title('KS ={}%'.format(int(KS*100)))
        plt.legend()
    return KS
# 画KS图
KS(df_test, 'score', 'target', plot = True)




test_df  = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')
def process(df_input):
    df=df_input.copy()
    df['Loan_Amount_Term'].fillna(Loan_Amount_Term_mean, inplace=True)
    # 对于离散型变量用众数填补
    df['Self_Employed'].fillna('No', inplace=True)
    df['Married'].fillna('Yes', inplace=True)
    df['Gender'].fillna('Male', inplace=True)
    df['Dependents'].fillna(0, inplace=True)
    df['Credit_History'].fillna(1, inplace=True)
    df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
    df['LoanAmount_log'] = np.log(df['LoanAmount'])
    for var in var_mod:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(df[var], prefix=var)
        df = df.join(cat_list)
    df_vars=df.columns.values.tolist()
    to_keep=[i for i in df_vars if i not in var_mod]
    df_final=df[to_keep]
    # 由于哑变量对每个离散型变量，产生k-1个哑变量
    var_delete = [ 'Gender_Male','Married_Yes','Dependents_0','Education_Not Graduate',
                   'Self_Employed_Yes',    'Property_Area_Rural', 'Loan_ID' ]
    for i in var_delete:
        del df_final[i]
    del  df_final['Loan_Amount_Term']
    del df_final['LoanAmount_log']
    return df_final

test_X=process(test_df)
       
test_Y = clf.predict(test_X)
test_df['Loan_status']=test_Y
test_df['Loan_status']=test_df['Loan_status'].map({1:'Y',0:'N'})
test_df.to_csv('result.csv')

