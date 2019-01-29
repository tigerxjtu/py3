#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: list_test.py
@time: 2019-01-09
'''

a=['wuchao','jinxin','xiaohu','sanpang','ligang',['wuchao','jinxin']]
#添加 append insert
a.append('xuepeng') #默认插到最后一个位置
a.insert(1,'xuepeng') #将数据插入到任意一个位置
#修改
a[1]='haidilao'
a[1:3]=['a','b']
#删除 remove pop del
a.remove(a[0])
b=a.pop(1)
print(a)
print(b)
del a[0]
del a
#a.remove(['wuchao','jinxin'])

#count:计算某元素出现次数
t=['to', 'be', 'or', 'not', 'to', 'be'].count('to')
#extend
a = [1, 2, 3]
b = [4, 5, 6]
a.extend(b)
print('extend==>',a)
# index # 根据内容找位置
a = ['wuchao', 'jinxin', 'xiaohu','ligang', 'sanpang', 'ligang', ['wuchao', 'jinxin']]
first_lg_index = a.index("ligang")
little_list = a[first_lg_index+1:]
second_lg_index = little_list.index("ligang")
second_lg_index_in_big_list = first_lg_index + second_lg_index +1
print(first_lg_index,little_list,second_lg_index,second_lg_index_in_big_list)
# reverse
a = ['wuchao', 'jinxin', 'xiaohu','ligang', 'sanpang', 'ligang']
a.reverse()
print('reverse==>',a)
x = [4, 6, 2, 1, 7, 9]
x.sort(reverse=True)
a = ['wuchao', 'jinxin', 'Xiaohu','Ligang', 'sanpang', 'ligang']
a.sort()
print('sort==>',a)