#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: lesson2.py
@time: 2018-09-27
'''

#1、写一个简单的python小程序，打印你的姓名，身高，年龄，城市；
print('姓名:%s, 身高:%.1f， 年龄:%d, 城市：%s'%('李寅',179.5,29,'上海'))

#2、将” lilei, hanmeimei, lucy,lily,jim”这个字符串用逗号,分割为5个字符串，并且去掉空格后保存在一个列表中，这些姓名的首字符要求大写；
s=" lilei, hanmeimei, lucy,lily,jim"
list=[item.strip().title() for item in s.split(',')]
print([list])

#3、对第2题得到的列表，增加你的姓名，并把你男神或者女神的名字添加到第一个位置；把’lily’修改成’Lemon’；用两种不同的方法删除掉’Lucy’这个姓名；
list.append('李寅')
list.insert(0,'习大大')
print(list)
#list[4]='Lemon'
list1=['Lemon' if item == 'Lily' else item for item in list]
print(list1)
#方法1
#del(list1[3])
#list1.remove('Lucy')
#方法2
list2=[i for i in filter(lambda x: x!='Lucy', list1)]
print(list2)

#4、把第3题得到的列表保存在一个元组中。
t=tuple(list2)
print(t)