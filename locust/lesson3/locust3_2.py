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
    @task
    def test_register(self):
        try:
            data = self.locust.user_data_queue.get()
            print('register with user: {}, pwd: {}' \
                  .format(data['phone'], data['passwd']))
            url = '/createUser?key=%s&phone=%s&passwd=%s' % (data['key'], data['phone'], data['passwd'])
            self.client.get(url)
        except queue.Empty:
            print('account data run out, test ended.')
            exit(0)



class WebsiteUser(HttpLocust):
    host = 'https://api.apiopen.top'
    task_set = UserBehavior
    user_data_queue = queue.Queue()

    headers=[]
    with open('data.csv', newline='') as csvfile:  # 此方法:当文件不用时会自动关闭文件
        csvReader = csv.reader(csvfile)
        for content in csvReader:
            if not headers:
                headers=content
                continue
            data = {}
            for i,key in enumerate(headers):
                data[key]=content[i]
            user_data_queue.put_nowait(data)

    min_wait = 1000
    max_wait = 3000

#
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