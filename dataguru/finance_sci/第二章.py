# -*- coding: utf-8 -*-
"""
This chapter two

@author: liyun
"""

#==============================================================================
# # 2.1 Python基本数据类型
#==============================================================================
# 2.1.1字符串
# Python中,单引号，双引号和三引号包围的都是字符串，如下所示：
a = 'spam egg'
b = "spam egg"
c = '''spam egg'''
d = str(1)
print( 'a 类型是 %s' % type(a))
print( 'b 类型是 %s' % type(b))
print( 'c 类型是 %s' % type(c))

#此外，Python中的字符串也支持一些格式化输出，例如换行符"\n"和制表符号 "\t"
print ('First line. \nSecond line.')
print ('1\t2')

#Python中，也会使用转义字符"\", 后一位的字符为原始输出.
print ("\"yes,\"he said.")

#也可以前面加r来表示原始输出
print('C:\Some\name')# 有换行符输出
print(r'C:\Some\name') #原始输出

#字符串支持拼接
a= 'pyt' + 'hon'
print (a)

# 2.1.2 浮点数和整数
# Python支持数值的四则运算，如下所示：
1 + 1 
5 - 3
1 * 1
2**2
2/3
5//2
5%4

#浮点数，其实在现实生活中，可以理解更加精确的数字，Python可以处理双精度浮点数，可以满足大部分数据分析的需求
# 可以使用内置函数进行转换
a = float(64)
b = int(a)
a=  float('1')
b = int('1')

# 2.1.3 布尔值
# Python布尔值一般通过逻辑判断产生,只有两个可能结果:True/False
1 == 1
1 > 3
'a' is 'a'

#在Python中，提供了逻辑值的运算，且，或和非的运算
True and False
True or False
not True

# 布尔逻辑值可以使用内置函数boo1，除数字0外，其他类型用bool转换结果都为True
bool(1)
bool(0)
bool("0")

# 2.1.4其他
# python中，还要一些特殊的数据类型。例如无穷值，nan(非数值),None等。
float('-inf')
float('inf')


float('-inf') + 1
float('inf')/-1

float('-inf') + float('inf')


#非数值nan在Python中与任何数值运算都会产生nan
#nan在Python中，可以表示空值
#此外，Python在提供了None表示为空，其仅仅支持判断运算，如下所示
x = None
x is None


#==============================================================================
# 2.2 Python数据结构
#==============================================================================
#2.2.1列表
a = [1,'2',3,4]
b = list([1,2,3])

#索引和切片
a[0] #访问第一个元素
a[-1] # 访问最后一个元素
a[:2] #第一个到第二个元素，开始位置包含，第三个不包含
a[0:]
a[::3] #第一位和第四位
a[::-1] # 倒序

#列表操作
a = [1,2,3]
b = [4,5,6]
a + b
c = list('python')
a + c 
a*3

#使用in判断元素是否在列表里面
1 in a
5 in a

#删除某个元素
del a[2]

#改变列表元素
a[2:5] = [3,4,5]
print(a)

#内置函数
max(a)
min(a)
len(a)


#列表的方法
a.append(6)
print(a)

a.extend([7,8,9])
print(a)
a.count(2)#统计2出现的次数
a.index(2) #找到2第一次出现的索引
a.insert(0,'Italy') #将对象插入列表中

a.pop() #移除最后一个元素
a.remove('Italy') #移除某个值第一个匹配项
a.reverse() #反向存放

a = [1,2,78,4,5,6]
a.sort() #对a排序

#2.2.2元组
#元组与列表类似，但元组中元素不可修改
#元组访问与列表一样
a = (1,2,3)
#元组访问与列表一样
a[0]
a[::-1]
a[2] = 4#修改则报错

#元组可以进行合并
a = (1,2,3)
b = (4,5,6,7)
c = a + b

#当对元组变量表达式进行赋值，会将右侧的值进行拆包复制给对应的对象，即元组拆包
a1,a2,a3,a4,a5,a6,a7 = c
print(a1,a3)


#2.2.3字典
#创建字典
d = {"Name":'Michael', 'Gender':'Male','Age':15, 'Height':68}
#访问字典的值
d['Name']

len(d)
d['city']= 'chengdu' #将值chengdu关联到键city上
del d['city'] #删除

d[23] ='Hello World'
#判断23是否在d的键中
23 in d
35 in d

#字典方法
#访问键gender的值

d.get("Gender")
d.get("gender") #访问键gender的值
d.get("gender",'1')

#将字典所有项以列表方式返回
d.items()
#返回d中的键
d.keys()
#返回d的值
d.values()


#2.2.4集合
a1 = set([1,2,3,1,4,5,3,7,8,9])
a2 = {1,1,3,3,4,4,3,3,5,6}

#a1和a2的差集(集合a1中去除a1和a2共有的元素)
a1 - a2
a1|a2 # a1和a2的并集
a1&a2 #a1和a2的交集
a1^a2 #即集合a1与a2的全部唯一元素去除集合a1和a2的公共元素

#集合常见方法
a1.add(10) #增加元素
a1.remove(10) #删除元素
{1}.issubset(a1) #判断1是否在a1中
a1.union(a2) #a1和a2并集
a1.intersection(a2)#a1和a2交集
a1.difference(a2) # a1 -a2
a1.symmetric_difference(a2) #a1^a2


#==============================================================================
# 2.3 Python程序结构
#==============================================================================
#2.3.1顺承结构
a = [1,2,3,4,5]
print (a[0])
print (a[1])
print (a[2])
print (a[3])
print (a[4])

#将逻辑行分为多个物理行
tuple(set(list([1,2,3,4,5,6,7,8])))
tuple(set(list([1,2,3,\
                4,5,6,7,8])))

#多个逻辑行过短时，可以转化为一个物理行
x=1; y=2;z=3
print(x,y,z)

#2.3.2 分支结构
x = -2
if x < 0:
    print ('Negative changed to zero')
elif x ==0: 
    print ('Zero')
elif x ==1:
    print('Single')
else:
    print ('More')
    

    
score = int(input('please input your score:'))
if score < 60:
    print ('成绩不合格')
elif 60 <=score < 70:
    print('成绩合格')
elif 70 <=score < 80:
    print ('成绩中')
elif 80 <=score < 90:
    print('成绩良')
else:
    print ('成绩优秀')

#2.3.3 循环结构
#while循环
#计算1到100的总和

x = 1
s = 0
while x <=100: 
    s +=x 
    x+=1 
print ("1到%d之和为: %d" %(x,s))

#while里面有else语句
count = 0
while count <5:
    print (count, 'is less than 5')
    count+=1 
else:
    print (count, 'is not less than 5, stop!')



x = 10
count = 0
while True:
    count = count +1
    x = x-0.02*x
    if x < 0.0001:
        break
print (x,count)


#比较break, continue和pass
count = 0
while count < 10:
    count =count+1
    if count % 3 ==0:
        print (count)
        continue


count = 0
while count < 10:
    count =count+1
    if count % 3 ==0:
        print (count)
        break

#pass语句
count = 0
while count < 10:
    count =count+1
    if count % 3 ==0:
        pass
    else:
        print(count)




#for 循环
a  =[1,2,3,4,5]
for i in a:
    print(i)

a = ['Mary','had','a','little','lamb']
for i in range(0,len(a)):
    print (i, a[i])

#遍历字典的值
d = {'name':'David','age':65,'gender':'male','department':'Statistics'}
for key in d:
    print (key,'corresponds to', d[key])


#计算100以内自然数之和
s = 0
for i in range(0,101):
    s+=i
print (s)

#乘法口诀表
for i in range(1,10):
    for j in range(1,i+1):
        print ("""%d*%d = %d""" %(i,j,i*j),end=" ")

# 列表推导式
a = [x*x for x in range(10)]



#==============================================================================
# 2.4  python的函数与模块
#==============================================================================
#2.4.1 python函数

def avg(x):
    mean_x = sum(x)/len(x)
    return mean_x

avg([23,24,13,34,56,78])


#计算一定范围内自然数之和
def snn(n,beg=1):
    s =  0
    for i in range(beg,n+1):
        s+=i
    return s

snn(1000) #计算100以内自然数之和

#匿名函数lambda
g = lambda x: x+1
g(10)
#相当于
def g(x):
    return (x+1)
g(1)


#调用函数对指定表达式进行操作
def universal(some_list,func):
    return [ func(x) for x in some_list]
a = [1,2,3,4,5]
b = universal(a,lambda x: x**2)
print (b)

#或者
g = lambda x: x**2
[g(x) for x in [1,2,3,4,5]]

#对指定表达式求值
b = universal(a,lambda x: x**3 -2*x**2-x)
#或者
g = lambda x: x**3 -2*x**2-x
[g(x) for x in [1,2,3,4,5]]



#2.4.2 python模块
#自定义模块
#掉用自定义模块（记住要把当前路径改为模块所在路劲)
import os
os.getcwd() #查看当前路劲
os.chdir("E:\Python培训(炼数成金)\课件\第二章") # 改为文件所在路径
import mod as m
a = [1,2,3,4,5]
m.mean(a)

#载入第三方库
import numpy as np
np.mean(a)
np.max(a)
np.min(a)
np.std(a)
np.median(a)


#==============================================================================
# 2.5  python读写数据
#==============================================================================
#读取csv文件
#导入相关文件
import pandas as pd
import os
os.chdir("E:\Python培训(炼数成金)\课件\第二章")
import numpy as np

sample = pd.read_csv('copyofhsb2.csv',index_col = 0) #用作行索引的列编号或者列名

#读取指定的列名和行数
sample = pd.read_csv('copyofhsb2.csv',usecols=['id','female'],nrows =2) #仅读取前两行

sample = pd.read_csv('copyofhsb2.csv',index_col = 0, na_values='70') #把70当成缺失值来处理

sample = pd.read_csv('copyofhsb2.csv',index_col = 0,encoding='utf-8') #用utf-8编码
#######pd.read_csv("copyofhsb2.csv",encoding="gb2312")

# 读取Excel文件
sample = pd.read_excel('E:\Python培训(炼数成金)\课件\第二章\\hsb2.xlsx',sheetname='hsb2',header=0)

sample = pd.read_excel('E:\Python培训(炼数成金)\课件\第二章\\hsb2.xlsx',sheetname='hsb2',\
                        dtype={'id':str,'female':np.str}) #将两个变量类型设置为字符串

#写出数据
#写出数据即保存数据
sample.to_csv('E:\Python培训(炼数成金)\课件\第二章\\a1.csv',index= False) #不写出索引列
sample.to_excel('E:\Python培训(炼数成金)\课件\第二章\\a1.xlsx',index= False) 
         
            
                                   
   

    






                       
                       
                       
        



