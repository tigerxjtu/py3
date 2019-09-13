# -*- coding: utf-8 -*-
"""
第二章数据处理部分

@author: liyun
"""

import pandas as pd
import numpy as np
import os

#==============================================================================
# 2.6数据处理
#==============================================================================
#==============================================================================
# #2.6.1 数据整合
#==============================================================================

#行列操作

#读取数据
os.chdir("E:\Python培训(炼数成金)\课件\第二章")
sample = pd.read_csv('copyofhsb2.csv',index_col = 0) #用作行索引的列编号或者列名
sample.head(5)
sample.tail(5)
#选择单列
sample['id']
sample.loc[:,'id']
sample[['id']][0:5]


#选择多行和多列
sample.ix[0:2,0:2] #不常用这种方法
sample.iloc[0:2,0:2]
sample.loc[0:2,['id','female']]
sample[['id','female']][:2]
sample.loc[0:6] #选择数据前6行
sample.iloc[0:6,:] #选择数据前6行

# 创建和删除列
sample['total_score'] = sample['math'] + sample['science']
sample = sample.assign(total_score1=sample['math'] + sample['science'],
              total_score2=sample['write'] + sample['read'])

#删除列
sample.drop('total_score',axis=1,inplace=True)
sample.drop(['total_score1','total_score2'],axis=1,inplace=True)




#条件查询
# 查找数学成绩及格的人
sample[sample.math >=60]
# or 
sample[sample['math'] >=60]

#多条件查询
sample[(sample['math'] >=60) & (sample['science'] >=60)] #且
sample[(sample['math'] >=60) | (sample['science'] >=60)] #或

#筛选出男性
sample[~(sample.female ==1)]

#使用loc按条件进行筛选

sample.loc[sample.math >=60,['id','female']]


#按照区间查找
sample[sample['math'].between(60,70,inclusive=True)]
#字符串按照isin来查找
sample[sample['id'].isin(['86','141'])] 
#匹配查找

sample =pd.DataFrame({'name':['Bob','Lindy','Mark',
                                     'Miki','Sully','Rose'],
						'score':[98,78,87,77,65,67],
						'group':[1,1,1,2,1,2],})
sample[sample['name'].str.contains('[M]+')]



# 横向连接
df1 = pd.DataFrame({'id':[1,2,3],
                    'col1':['a','b','c']})
df2 = pd.DataFrame({'id':[4,3],
                    'col2':['d','e']})

#内连接
df3  = df1.merge(df2,how='inner',on = 'id')
df3  = df1.merge(df2,how='inner',left_on = 'id',right_on ='id')
#或者

df3 = pd.merge(df1,df2,how='inner',on = 'id')
df3 = pd.merge(df1,df2,how='inner',left_on = 'id',right_on ='id')


#外连接
df3 = pd.merge(df1,df2,how='left',on='id') #左连接
df3 = pd.merge(df1,df2,how='right',on='id') #右连接
df3 = pd.merge(df1,df2,how='outer',on='id') #全连接


#纵向连接
df1 = pd.DataFrame({'id':[1,1,1,2,3,4,6],
                    'col':['a','a','b','c','v','e','q']})
df2 = pd.DataFrame({'id':[1,2,3,3,5],
                    'col':['x','y','z','v','w']})

df3 = pd.concat([df1,df2],ignore_index=True,axis=0)

#排序
#按照升序排
sample = pd.read_csv('copyofhsb2.csv',index_col = 0) #重新读取
sample.sort_values('math',ascending = True,na_position='last',inplace=True) #升序
sample.sort_values('math',ascending = False,na_position='last',inplace=True) #降序


#也可以根据多个变量排名
sample = sample.sort_values(['female','math'])


#分组汇总
sample = pd.read_csv('sample.csv',encoding='gbk')
#按照年级查找数学成绩最高值
sample.groupby('grade')[['math']].max()
#按照年级和班级进行数学成绩汇总统计
sample.groupby(['grade','class'])[['math']].max()
sample.groupby(['grade','class'])[['math']].mean()

#汇总变量
# 计算每个年级，数学和语文的平均分
sample.groupby(['grade'])['math','chinese'].mean()


sample.groupby(['class'])['math','chinese'].agg(['mean','min','max'])


#拆分和堆叠列
table = pd.DataFrame({'cust_id':[10001,10001,10002,10002,10003],
                      'type':['Normal','Special_offer',\
                              'Normal','Special_offer','Special_offer'],
                      'Monetary':[3608,420,1894,3503,4567]})


table1 = pd.pivot_table(table,index='cust_id',columns='type',values='Monetary').reset_index()

#堆叠列
table2= pd.melt(table1,id_vars='cust_id',value_vars=['Normal','Special_offer'],
                value_name='Monetary',var_name='TYPE')



##赋值与条件赋值
sample = pd.DataFrame({'name':['Bob','Lindy','Mark',
		 'Miki','Sully','Rose'],
		'score':[99,78,999,77,77,np.nan],
		'group':[1,1,1,2,1,2],})

#将999替换成缺失
sample.score.replace(999,np.nan,inplace=True)

#多个值变量替换
sample.replace({'score':{999:np.nan},'name':{'Bob':np.nan}},inplace=True)


#条件赋值
def tran(row):
    if row['group'] ==1:
        return ('class1')
    else:
        return ('class2')

sample.apply(tran,axis=1)

sample.assign(class_n=sample.apply(tran,axis=1))

#增加新的一列
sample['class_n'] = None
sample.loc[sample.group ==1,'class_n'] = 1
sample.loc[sample.group ==2,'class_n'] = 2


#==============================================================================
# 2.6.2 数据清洗
#==============================================================================
#重复值处理
sample = pd.DataFrame({'id':[1,1,1,3,4,5],
                       'name':['Bob','Bob','Mark','Miki','Sully','Rose'],
                       'score':[99,99,87,77,77,np.nan],'group':[1,1,1,2,1,2]})
#查看重复数据
sample[sample.duplicated()] 
sample[sample['id'].duplicated()]

#剔除重复数据
sample.drop_duplicates(inplace=True) #按照所有变量
sample.drop_duplicates('id',inplace=True) #按照ID来去除


#缺失值处理
sample = pd.DataFrame({'id':[1,1,np.nan,3,4,np.nan],
                       'name':['Bob','Bob','Mark','Miki','Sully',np.nan],
                       'score':[99,99,87,77,77,np.nan],'group':[1,1,1,2,np.nan,2]})

#查看缺失情况
sample.isnull()
sample.apply(lambda x: sum(x.isnull())) #统计每列缺失个数
sample.apply(lambda x: sum(x.isnull())/len(x)) #统计每列缺失比例

#缺失值填补
sample['score'].fillna(sample.score.mean(),inplace=True)
sample.fillna(-1,inplace=True) #用-1填补缺失值


#删除法
sample.dropna(axis =0,how='any',inplace = True) #1代表列，0代表行
sample.dropna(axis =1,how='all',inplace = True) #1代表列，0代表行
sample.dropna(axis =0,how='any',subset=['group','id'],inplace = True) #1代表列，0代表行


#==============================================================================
# 2.6.3 时序时间处理
#==============================================================================
#时间/日期数据转换

#时间与日期
from datetime import datetime
dt =  datetime(2018,3,20,10,8,2)
dt1 = datetime(2018,4,20,8,8,2)
dt1 - dt 

dt2 = dt.strftime('%m/%d/%Y %H:%M:%S') #转化为字符串格式

jd = pd.read_excel('E:\Python培训(炼数成金)\课件\第二章\\data.xlsx',header= None,dtype=object)
jd['time'] =pd.to_datetime(jd[1]) 

jd['time'].max()
jd['time'].min()

#时序数据基础操作

#提取时间序列相关信息
year1 = [i.year for i in jd['time']]
month1 = [i.month for i in jd['time']]
week1 = [i.week for i in jd['time']]

#加减时间
jd['time1'] = jd['time'] + pd.Timedelta(days=1)

jd['timedelta'] = jd['time'] -  pd.to_datetime('2017-1-1')
jd['timedelta'] = jd['timedelta']/np.timedelta64(1, 'D')   #转化为天数


jd['timedelta'].max()
jd['timedelta'].min()
jd['timedelta'].mean()



















