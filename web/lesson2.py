#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: lesson2.py
@time: 2018-10-22
'''

'''
定义一个函数，判断函数的输入是否为合法的IP地址（ipv4中规定的IP地址），例如 114.23.78.45 就是一个合法的IP地址，关于IP地址的正确格式大家可在网络查询。
函数的要求：
1、函数声明格式必须为 def is_ipv4(istr):
2、输入若为合法的IP地址，返回True，否则返回False
3、函数内容使用本课知识编程实现，不要使用正则判断。
将编程结果以 “学号.py” 的文件形式上传，不要粘贴到网页中。若系统不允许上传py后缀的文件，可以将“学号.py” 打包为“学号.zip”进行上传。 “学号.py” 中可以写自己的测试代码，但不要影响is_ipv4函数的调用。
'''

def is_ipv4(istr):
    parts=istr.split('.')
    if len(parts)!=4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        i=int(part)
        if i<0 or i>255:
            return False
    return True

ips=['127.0.0.1','226.2.1.23','223.21.4','127.0.0.a','112.234.23.256']
for ip in ips:
    print(ip,is_ipv4(ip))
