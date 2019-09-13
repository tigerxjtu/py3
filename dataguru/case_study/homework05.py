# -*- coding: utf-8 -*-

#数据读入
import pandas as pd
import numpy as np

import re 

path=u'D:/data/example05/spider/用户45.xls'
user=pd.read_excel(path,sheetname=0,header=1)
user.head()

#统计发言时间
def get_times(path):
    import pandas as pd
    from datetime import datetime
    from datetime import date 
    user=pd.read_excel(path,sheetname=0,header=1)

    #统计发言时间   
    time1=user[u'/基本信息/item/时间']
    pattern=re.compile(r'\d{4}-\d+-\d+ \d{2}:\d{2}')
    time_1=[]
    for time in time1:
        if type(time)==unicode:
            res=pattern.match(time)
            if res:
                time_1.append(res.group())

    times=[]
    for time in time_1:
    #将发言时间转化为datetime格式
        times.append(datetime.strptime(time,"%Y-%m-%d %H:%M"))  
         
    weekdays=[]
    for time in times:
        weekdays.append(date.isoweekday(time))  
    weekdays_count=pd.Series(weekdays).value_counts()

    hours=[]
    for time in times:
        hours.append(time.strftime('%H'))
    hours_count=pd.Series(hours).value_counts() 
    
    return weekdays_count , hours_count
    
weekday,hour=get_times(path)

#读入类别数据
sample = pd.read_excel(u'd:/data/example05/result2.xls')
sample = sample.as_matrix() 

#构建分类模型
from sklearn.tree import DecisionTreeClassifier #导入决策树模型
tree = DecisionTreeClassifier() #建立决策树模型
tree.fit(sample[:,:18], sample[:,18]) #训练

tree.predict(hour)