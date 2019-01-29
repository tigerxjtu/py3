#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: locust3_1.py
@time: 2019-01-21
'''

# https://www.apiopen.top/createUser?key=00d91e8e0cca2b76f515926a36db68f5&phone=13594347817&passwd=123654

import queue
from locust import TaskSet, task, HttpLocust
import csv

class UserBehavior(TaskSet):
    def on_start(self):
        self.index = 0

    @task
    def test_visit(self):
        data=self.locust.share_data[self.index]
        url = '/createUser?key=%s&phone=%s&passwd=%s'%(data['key'],data['phone'],data['passwd'])
        print('visit url: %s' % url)
        self.index = (self.index + 1) % len(self.locust.share_data)
        self.client.get(url)


class WebsiteUser(HttpLocust):
    host = 'https://api.apiopen.top'
    task_set = UserBehavior
    share_data = []
    with open('data.csv', newline='') as csvfile:  # 此方法:当文件不用时会自动关闭文件
        csvReader = csv.reader(csvfile)
        headers = []
        for content in csvReader:
            print(content)
            if not headers:
                headers = content
                # print(header)
                continue
            data = {}
            for i, key in enumerate(headers):
                data[key] = content[i]
            share_data.append(data)
    min_wait = 1000
    max_wait = 3000


# if __name__=='main':
#     print('start')
#     with open('data.csv', newline='') as csvfile:  # 此方法:当文件不用时会自动关闭文件
#         csvReader = csv.reader(csvfile)
#         headers = []
#         for content in csvReader:
#             print(content)
#             if not headers:
#                 headers=content
#                 # print(header)
#                 continue
#             data = {}
#             for i,key in enumerate(headers):
#                 data[key]=content[i]
#             print(data)