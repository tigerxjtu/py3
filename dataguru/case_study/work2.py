# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:55:58 2017

@author: tiger
"""

import numpy as np
import pandas as pd
import xlrd, xlwt
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering



def sumRow(row, collection=(4,5,12,13,14)):
    inset=0
    notin=0
    for i in range(len(row)):
        if i in collection:
            inset+=row[i]
        else:
            notin+=row[i]
    return (inset,notin)
    

dir=u'D:/data/example02/2015xxxx/%d.xls'
files=range(1,11)
pds=[]
columns=[u'上下班时间段进站日均人数',u'非上下班时间段进站日均人数',u'上下班时间段出站日均人数',u'非上下班时间段出站日均人数']
for i in files:
    filename=dir%i
    book=xlrd.open_workbook(filename)
    sheet = book.sheet_by_index(0)
    index=sheet.col_values(0)[1:-2]
    index=index[::2]
    #columns=sheet.row_values(0,start_colx=2,end_colx=26)
    values=[]
    lastRow=False
    for r in range(1,sheet.nrows-2):
        if lastRow:
            index=index[:-1]
            break
        row=sheet.row_values(r,start_colx=2,end_colx=26)
        t=sumRow(row)
        if r%2==1:
            record=[]
        record.extend([t[0],t[1]])
        if r%2==0:
            values.append(record)
            if r in (sheet.nrows-4,sheet.nrows-3):
                lastRow=True
    df=pd.DataFrame(np.array(values,dtype=np.float),columns=columns,index=index)
    pds.append(df)
data=pd.concat(pds)

data = (data - data.min())/(data.max() - data.min()) #离差标准化
data=data.fillna(0)  #处理na值

#dataframe.describe()
Z = linkage(data, method = 'ward', metric = 'euclidean') #谱系聚类图
P = dendrogram(Z, 0) #画谱系聚类图
plt.show()

k = 4 #聚类数
model = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
model.fit(data) #训练模型
data[u'聚类类别']=pd.Series(model.labels_, index = data.index)

style = ['ro-', 'go-', 'bo-', 'ko-']
xlabels = columns
pic_output = 'd:/data/example02/work_' #聚类图文件名前缀

for i in range(k): #逐一作图，作出不同样式
  plt.figure()
  tmp = data[data[u'聚类类别'] == i].iloc[:,:4] #提取每一类
  for j in range(len(tmp)):
    plt.plot(range(1, 5), tmp.iloc[j], style[i])
  
  plt.xticks(range(1, 5), xlabels, rotation = 20,fontproperties='SimHei') #坐标标签
  plt.title(u'商圈类别%s' %(i+1),fontproperties='SimHei') #我们计数习惯从1开始
  plt.subplots_adjust(bottom=0.15) #调整底部
  plt.savefig(u'%s%s.png' %(pic_output, i+1)) #保存图片
            
        
dir=u'D:/data/example02/2015xxxx/%d.xls'
files=range(0,12)
print files[::2]
pds=[]
columns=[u'上下班时间段进站日均人数',u'非上下班时间段进站日均人数',u'上下班时间段出站日均人数',u'非上下班时间段出站日均人数']
#for i in files:
filename=dir%1
book=xlrd.open_workbook(filename)
sheet = book.sheet_by_index(0)
index=sheet.col_values(0)[1:-2]
print index
index=index[::2]
print index
#columns=sheet.row_values(0,start_colx=2,end_colx=26)
values=[]
for r in range(1,sheet.nrows-3):
    #print len(sheet.row_values(r,start_colx=2,end_colx=26))
    row=sheet.row_values(r,start_colx=2,end_colx=26)
    t=sumRow(row)
    if r%2==1:
        record=[]
    record.extend([t[0],t[1]])
    if r%2==0:
        values.append(record)
df=pd.DataFrame(np.array(values,dtype=np.float),columns=columns,index=index)
print df.head() 
df      
        



l=range(2,27)
print l[6]

for i in files:
    filename=dir%i
    #print filename
    data=pd.read_excel(filename,skip_footer=3)
    data.drop(data.columns[-1],axis=1,inplace=True)
    print data.tail()
    
filename=dir%10
book=xlrd.open_workbook(filename)
sheet = book.sheet_by_index(0)
columns=sheet.row_values(0,start_colx=2,end_colx=26)
print sheet.row_values(2,start_colx=2,end_colx=26)
index=sheet.col_values(0)[1:-3]
index=index[::2]
print columns
print index



#print filename
data=pd.read_excel(filename,skip_footer=3)

data.drop(data.columns[-1],axis=1,inplace=True)
print data.tail()
