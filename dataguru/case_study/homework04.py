# -*- coding: utf-8 -*-

###数据探索分析###
import pandas as pd

datafile= 'd:/data/air_data.csv' #航空原始数据,第一行为属性标签
resultfile = 'd:/data/explore.xls' #数据探索结果表

data = pd.read_csv(datafile, encoding = 'utf-8') #读取原始数据，指定UTF-8编码（需要用文本编辑器将数据装换为UTF-8编码）

explore = data.describe(percentiles = [], include = 'all').T 
#包括对数据的基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）；T是转置，转置后更方便查阅
explore
explore['null'] = len(data)-explore['count'] #describe()函数自动计算非空值数，需要手动计算空值数

explore = explore[['null', 'max', 'min']]
explore.columns = [u'空值数', u'最大值', u'最小值'] #表头重命名
explore
explore.to_excel(resultfile) #导出结果

###数据预处理####

datafile= 'd:/data/air_data.csv' #航空原始数据,第一行为属性标签
cleanedfile = 'd:/data_cleaned.xls' #数据清洗后保存的文件

data = pd.read_csv(datafile,encoding='utf-8') #读取原始数据，指定UTF-8编码（需要用文本编辑器将数据装换为UTF-8编码）

data = data[data['SUM_YR_1'].notnull()*data['SUM_YR_2'].notnull()] #票价非空值才保留

#只保留票价非零的，或者平均折扣率与总飞行公里数同时为0的记录。
index1 = data['SUM_YR_1'] != 0
index2 = data['SUM_YR_2'] != 0
index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0) #该规则是“与”
data = data[index1 | index2 | index3] #该规则是“或”

data_2=pd.DataFrame()

data_2['L'] = data['LOAD_TIME'] - data['FFP_DATE']
data_2['R'] = data['LAST_TO_END']
data_2['F'] = data['FLIGHT_COUNT']
data_2['M'] = data['SEG_KM_SUM']
data_2['C'] = data['avg_discount']


data_2.to_excel(cleanedfile) #导出结果


#标准化处理
data_3 = (data_2 - data_2.mean(axis = 0))/(data_2.std(axis = 0)) #简洁的语句实现了标准化变换，类似地可以实现任何想要的变换。
data_3.columns=['Z'+i for i in data_2.columns] #表头重命名。

###模型构建###
from sklearn.cluster import KMeans #导入K均值聚类算法
k = 5                       #需要进行的聚类类别数

#读取数据并进行聚类分析
data = data_3 #读取数据

#调用k-means算法，进行聚类分析
kmodel = KMeans(n_clusters = k, n_jobs = 4) #n_jobs是并行数，一般等于CPU数较好
kmodel.fit(data) #训练模型

kmodel.cluster_centers_ #查看聚类中心
kmodel.labels_ #查看各样本对应的类别