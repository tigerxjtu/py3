#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: locust1.py
@time: 2019-01-14
'''

from locust import HttpLocust, TaskSet, task


class WebsiteTasks(TaskSet):

    # def on_start(self):
    #     self.client.post("/login", {
    #         "username": "test",
    #         "password": "123456"
    #     })

    @task(5)
    def email(self):
        self.client.get("/EmailSearch?number=1012002")

    @task(3)
    def news(self):
        self.client.get('/journalismApi')

    @task(1)
    def music(self):
        self.client.get("/musicRankingsDetails?type=1")


class WebsiteUser(HttpLocust):
    task_set = WebsiteTasks
    host = "https://api.apiopen.top"
    min_wait = 1000
    max_wait = 5000
