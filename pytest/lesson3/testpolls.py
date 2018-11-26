#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: testpolls.py
@time: 2018-09-29
'''

import unittest
import json
import requests

class Lesson3Test(unittest.TestCase):

    def setUp(self):
        self._url="http://127.0.0.1:8000"

    def get_url(self, url):
        r=requests.get(url)
        return (r.status_code, json.loads(r.text))

    def post_url(self,url,params):
        r=requests.post(url,data=params)
        return (r.status_code, json.loads(r.text))

    def test_polls(self):
        url=self._url+"/polls/2/"
        code,obj=self.get_url(url)
        self.assertEqual(code,200)
        self.assertEqual(obj['status'],'200')
        self.assertEqual(obj['data']['4'], '泰国')

    def test_vote(self):
        url=self._url+"/polls/2/vote/"
        code,obj=self.post_url(url,{'choice':'4'})
        self.assertEqual(code,200)
        self.assertEqual(obj['status'],'200')
        self.assertEqual(obj['data']['question'], '你喜欢去哪里旅游？')


if __name__ == '__main__':
    unittest.main()