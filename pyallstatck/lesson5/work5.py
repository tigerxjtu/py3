#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: work5.py
@time: 2018-10-23
'''


'''
复习及练习面向对象，要求自定义几个类，把今天学习的封装、继承、多态、多重继承。
'''

class Base():
    def __init__(self):
        self.name='Base'

    def print_info(self):
        print('My name is ',self.name)

    def do(self):
        print('do in Base')

class Derived1(Base):
    def __init__(self,name):
        self.name=name

    def do(self):
        print('do in Derived1')

class Another():
    def another(self):
        print('another method')

class Derived2(Base,Another):
    def __init__(self,name):
        self.name=name

    def do(self):
        print('do in Derived2')

print('封装')
base1=Base()
base1.print_info()
print('继承')
derived1=Derived1('Derived1')
base1.print_info()
derived1.print_info()
print('多重继承')
derived2=Derived2('Derived2')
derived2.print_info()
derived2.another()

print('多态')
def do_something(base):
    base.do()

do_something(derived1)
do_something(derived2)



