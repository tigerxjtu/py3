#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: testvote.py
@time: 2018-10-06
'''

from pytest.lesson4.testrequest import *

_url="http://127.0.0.1:8000"


def test_polls():
    url = _url + "/polls/2/"
    result=TestGetRequest(url, {}, "lesson4-1", "测试投票主题", "200", "泰国")
    print(result)


def test_vote():
    url = _url + "/polls/2/vote/"
    result = TestPostRequest(url, {'choice': '4'}, "lesson4-2", "测试明细", "200", "你喜欢去哪里旅游",None)
    print(result)


def test_login():
    url = _url + "/polls/login/"
    result = TestPostRequest(url, {'username': 'abc', 'password': '123456'}, "lesson4-3", "测试登录", "200", "success",None)
    print(result)

if __name__=="main":
    test_polls()
    test_vote()
    test_login()
