#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: relation.py
@time: 2019-01-29
'''
import re
from locust import TaskSet, task, HttpLocust

class UserBehavior(TaskSet):
    @staticmethod
    def get_key(html):
        regexp = r'.*{"key":"(?P<key>[a-z0-9]+)".*'
        reg = re.compile(regexp)
        m=reg.match(html)
        key=m.group('key')
        print(key)
        return key


    @task(10)
    def test_login(self):
        # https://www.apiopen.top/login?key=00d91e8e0cca2b76f515926a36db68f5&phone=13594347817&passwd=123456
        username='13994367817'
        passwd='123654'
        html = self.client.get('/login?key=00d91e8e0cca2b76f515926a36db68f5&phone=%s&passwd=%s'%(username,passwd)).text
        assert '"code":200' in html, 'response error:'+html
        key = self.get_key(html)
        self.client.get('/findStatistics?key=%s'%key)

class WebsiteUser(HttpLocust):
    host = 'https://www.apiopen.top'
    task_set = UserBehavior
    min_wait = 1000
    max_wait = 3000