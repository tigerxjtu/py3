# -*- coding: utf-8 -*-
#加载包
import pandas as pd
from sqlalchemy import create_engine

#连接数据库
engine = create_engine('mysql+pymysql://root:dataguru@127.0.0.1:3306/mysql?charset=utf8')
sql = pd.read_sql('all_gzdata', engine, chunksize = 10000)

###数据预处理###
#数据清洗
for i in sql:
  d = i[['realIP', 'fullURL']] #只要网址列
  d = d[d['fullURL'].str.contains('\.html')].copy() #只要含有.html的网址
  #保存到数据库的cleaned_gzdata表中（如果表不存在则自动创建）
  d.to_sql('cleaned_gzdata', engine, index = False, if_exists = 'append')
  
#数据变换
sql = pd.read_sql('cleaned_gzdata', engine, chunksize = 10000)
for i in sql: #逐块变换并去重
  d = i.copy()
  d['fullURL'] = d['fullURL'].str.replace('_\d{0,2}.html', '.html') #将下划线后面部分去掉，规范为标准网址
  d = d.drop_duplicates() #删除重复记录
  d.to_sql('changed_gzdata', engine, index = False, if_exists = 'append') #保存

#网站分类
sql = pd.read_sql('changed_gzdata', engine, chunksize = 10000)
for i in sql: #逐块变换并去重
  d = i.copy()
  d['type_1'] = d['fullURL'] #复制一列
  d['type_1'][d['fullURL'].str.contains('(ask)|(askzt)')] = 'zixun' #将含有ask、askzt关键字的网址的类别一归为咨询（后面的规则就不详细列出来了，实际问题自己添加即可）
  d.to_sql('splited_gzdata', engine, index = False, if_exists = 'append') #保存


###模型构建###
import numpy as np

def Jaccard(a, b): #自定义相似系数
  return 1.0*(a*b).sum()/(a+b-a*b).sum()

class Recommender():
  
  sim = None #相似度矩阵
  
  def similarity(self, x): #计算相似度矩阵的函数
      return x.corr()
  
  def fit(self, x, distance = Jaccard): #训练函数
    self.sim = self.similarity(x)
  
  def recommend(self, a): #推荐函数
    return np.dot(self.sim, a)*(1-a)  


sql = pd.read_sql('splited_gzdata', engine, chunksize = 10000)      
data = [d[d['type_1']=='zixun'][['realIP', 'fullURL']] for d in sql]
data = pd.concat(data).iloc[:100,:100]

urls=set(data['fullURL'])
users=set(data['realIP'])
m=len(urls)
n=len(users)

input_urls=list(urls)
input_users=list(users)
input_data=np.zeros((n,m))

for i in range(n):
    for j in range(m):
        if(input_urls[j] in set(data[data['realIP']==input_users[i]]['fullURL'])):
            input_data[i,j]=1

rec=Recommender()
rec.fit(input_data.T)
res=rec.recommend(input_data[0])
