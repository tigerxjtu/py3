# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:46:23 2017

@author: tiger
"""
'''
CREATE TABLE ip_cnt AS 
SELECT realip,COUNT(DISTINCT fullurl) cnt 
FROM all_gzdata GROUP BY realip ORDER BY cnt DESC LIMIT 1000

CREATE TABLE url_cnt AS 
SELECT fullurl,COUNT(DISTINCT realip) cnt 
FROM all_gzdata GROUP BY fullurl ORDER BY cnt DESC LIMIT 1000

CREATE TABLE ip_url AS SELECT a.realip,a.fullurl FROM all_gzdata a,ip_cnt i, url_cnt u 
WHERE a.realip=i.realip AND a.fullurl=u.fullurl
'''
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def Jaccard(a, b): #自定义相似系数
  return 1.0*(a*b).sum()/(a+b-a*b).sum()

class Recommender():
  
  sim = None #相似度矩阵
  
  def similarity(self, x, distance): #计算相似度矩阵的函数
    y = np.ones((len(x), len(x)))
    for i in range(len(x)):
      for j in range(len(x)):
        y[i,j] = distance(x[i], x[j])
    return y
  
  def fit(self, x, distance = Jaccard): #训练函数
    self.sim = self.similarity(x, distance)
  
  def recommend(self, a): #推荐函数
    return np.dot(self.sim, a)*(1-a)  

#连接数据库
engine = create_engine('mysql+pymysql://root:@192.168.1.21:3306/jfedu?charset=utf8')

sql = pd.read_sql('ip_url', engine, chunksize = 10000)      
data = [d[['realip', 'fullurl']] for d in sql]
#data = pd.concat(data).iloc[:100,:100]
data = pd.concat(data)

urls=set(data['fullurl'])
users=set(data['realip'])
m=len(urls)
n=len(users)

input_urls=list(urls)
input_users=list(users)
input_data=np.zeros((n,m))

for i in range(n):
    for j in range(m):
        if(input_urls[j] in set(data[data['realip']==input_users[i]]['fullurl'])):
            input_data[i,j]=1
#for i in xrange(len(data)):
    

rec=Recommender()
rec.fit(input_data.T)
res=rec.recommend(input_data[0])
uis=[i for (i,d) in enumerate(input_data[0]) if d==1]
items=[(i,v) for (i,v) in enumerate(res) if v>=1]
print items
recommend_urls=[input_urls[i] for (i,v) in items ]
print recommend_urls
'''
print recommend_urls
[u'http://www.lawtime.cn/mylawyer/index.php?m=case&a=share', u'http://www.lawtime.cn/mylawyer/index.php?m=ask&a=zhuiwen', u'http://www.lawtime.cn/guangzhou', u'http://www.lawtime.cn/mylawyer/index.php?m=case&a=index', u'http://www.lawtime.cn/mylawyer/index.php?m=msg&a=system', u'http://www.lawtime.cn/mylawyer/index.php?m=case&a=index&page=4', u'http://www.lawtime.cn/mylawyer/index.php?m=case&a=index&page=7', u'http://www.lawtime.cn/mylawyer/index.php?m=case&a=index&page=3', u'http://www.lawtime.cn/mylawyer/index.php?m=site&a=index', u'http://www.lawtime.cn/mylawyer/index.php?m=case&a=index&page=2', u'http://www.lawtime.cn/mylawyer/index.php?m=online&a=zhuiwen', u'http://www.lawtime.cn/mylawyer/index.php?m=info&a=editinfo', u'http://www.lawtime.cn/mylawyer/index.php?m=article&a=index', u'http://www.lawtime.cn/guangzhou/lawyer/p1ll110801', u'http://www.lawtime.cn/mylawyer/index.php?m=msg&a=usermsg', u'http://www.lawtime.cn/mylawyer/index.php?m=article&a=fabu', u'http://www.lawtime.cn/mylawyer/index.php?m=ask', u'http://www.lawtime.cn/info/xingfa/', u'http://www.lawtime.cn/mylawyer/index.php?m=case&a=myArea']
'''

