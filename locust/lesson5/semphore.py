#!/usr/bin/python3
# encoding:utf-8

import re
from locust import HttpLocust, TaskSet, task, events
from gevent._semaphore import Semaphore
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
        key = self.get_key(html)
        self.client.get('/findStatistics?key=%s' % key)

    def on_start(self):
        all_locusts_spawned.wait()

class WebsiteUser(HttpLocust):
    host = 'https://www.apiopen.top'
    task_set = UserBehavior
    min_wait = 100
    max_wait = 1000