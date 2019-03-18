#!/usr/bin/python3
# encoding:utf-8

import re
from locust import HttpLocust, TaskSet, task, events
from gevent._semaphore import Semaphore
import queue
import csv

all_locusts_spawned = Semaphore()
all_locusts_spawned.acquire()

def on_hatch_complete(**kwargs):
    all_locusts_spawned.release()

events.hatch_complete += on_hatch_complete

class UserBehavior(TaskSet):

    @staticmethod
    def get_key(html):
        regexp = r'.*{"key":"(?P<key>[a-z0-9]+)".*'
        reg = re.compile(regexp)
        m = reg.match(html)
        key = m.group('key')
        return key

    @task(10)
    def test_login(self):
        # https://www.apiopen.top/login?key=00d91e8e0cca2b76f515926a36db68f5&phone=13594347817&passwd=123456
        username = '13994367817'
        passwd = '123654'
        html = self.client.get(
            '/login?key=00d91e8e0cca2b76f515926a36db68f5&phone=%s&passwd=%s' % (username, passwd)).text
        assert '"code":200' in html, 'response error:' + html
        key = self.get_key(html)
        self.client.get('/findStatistics?key=%s' % key)

    @task(10)
    def test_register(self):
        try:
            data = self.locust.user_data_queue.get()
            print('register with user: {}, pwd: {}' \
                  .format(data['phone'], data['passwd']))
            url = '/createUser?key=%s&phone=%s&passwd=%s' % (data['key'], data['phone'], data['passwd'])
            self.client.get(url)
            self.locust.user_data_queue.put_nowait(data)
        except queue.Empty:
            print('account data run out, test ended.')
            exit(0)

    @task(5)
    def email(self):
        self.client.get("/EmailSearch?number=1012002")

    @task(5)
    def news(self):
        self.client.get('/journalismApi')

    @task(2)
    def music(self):
        self.client.get("/musicRankingsDetails?type=1")

    def on_start(self):
        all_locusts_spawned.wait()

class WebsiteUser(HttpLocust):
    host = 'https://www.apiopen.top'
    task_set = UserBehavior

    user_data_queue = queue.Queue()

    headers = []
    with open('data.csv', newline='') as csvfile:  # 此方法:当文件不用时会自动关闭文件
        csvReader = csv.reader(csvfile)
        for content in csvReader:
            if not headers:
                headers = content
                continue
            data = {}
            for i, key in enumerate(headers):
                data[key] = content[i]
            user_data_queue.put_nowait(data)

    min_wait = 100
    max_wait = 1000