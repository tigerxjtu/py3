#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: student.py
@time: 2018-10-18
'''


# def student_info(code,name,*score,major='计算机科学与技术',college='软件学院',**kw):
#     print('学号：{}'.format(code))
#     print('姓名：{}'.format(name))
#     print('学院：{}'.format(college))
#     print('专业：{}'.format(major))
#     for k in kw:
#         print('{}：{}'.format(k,kw[k]))
#     print(len(score))
#     print('各科成绩总分为：{}， 平均分：{}'.format(sum(score), sum(score)/len(score)))

def student_info(code,name,*score,major='计算机科学与技术',college='软件学院',**kw):
    print('学号：{}'.format(code))
    print('姓名：{}'.format(name))
    print('学院：{}'.format(college))
    print('专业：{}'.format(major))
    # for k in kw:
    #     print('{}：{}'.format(k,kw[k]))
    if 'sex' in kw:
        print('{}：{}'.format('性别', kw['sex']))
    if 'age' in kw:
        print('{}：{}'.format('年龄', kw['age']))
    if 'city' in kw:
        print('{}：{}'.format('籍贯', kw['city']))
    if len(score)>0:
        print('各科成绩总分为：{}， 平均分：{}'.format(sum(score), sum(score)/len(score)))

info={'sex':'男','age':19}
score=(90,80,95)
student_info('001','张三',*score,**info)

info={'sex':'女','age':18,'city':'上海'}
score=(80,85,75,80)
student_info('002','李四',*score,college='网络安全学院',major='计算机软件',**info)


score=(80,85,75,80)
student_info('003','王五',*score,major='计算机软件',age=18,city='北京')
