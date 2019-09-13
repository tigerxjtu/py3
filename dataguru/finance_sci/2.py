# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:13:22 2018

@author: DELL
"""

#Q1. 生产一个list,包含20以内的奇数，并用两种方法访问新生成序列的最后5个数据
a=[i for i in range(1,21) if i%2==1]
print(a)

print(a[-5:]) #方法1
print(a[len(a)-5:]) #方法2

#Q2.  (1) 生成如下图所示的DataFrame, 并且输出第三行。
import pandas as pd

data={'a':['Alice',34],
      'b':['Bob',36],
      'c':['Charlie',30],
      'd':['David',29],
      'e':['Esther',32],
      'f':['Fanny',36]}

df=pd.DataFrame.from_dict(data,orient='index')
df.columns=['name','age']
print(df)
df1=df.reset_index()
df1.columns=['id','name','age']
print(df1)

print(df1.iloc[2])
#     (2) 在(1)的基础上新增加”g,John,19”一行,将年龄设置为索引，删除第三行后输出结果
df2=pd.DataFrame([['g','John',19]],columns=['id','name','age'])
df3=df1.append(df2,ignore_index=True)
df4=df3.set_index(['age'])
df4.drop([30],axis=0,inplace=True)
print(df4)

