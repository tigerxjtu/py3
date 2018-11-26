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


class UserTest(unittest.TestCase):

    def setUp(self):
        self._url = "http://127.0.0.1:8000"

    def get_url(self, url, params):
        r = requests.get(url, params=params)
        return r.status_code, json.loads(r.text)

    def post_url(self, url, params):
        r = requests.post(url, data=params)
        return r.status_code, json.loads(r.text)

    def test_reg(self):
        url = self._url + "/polls/reg/"
        code, obj = self.get_url(url, {'username': 'abc', 'password': '123456'})
        self.assertEqual(code, 200)
        self.assertEqual(obj['status'], '200')

    def test_login(self):
        url = self._url + "/polls/login/"
        code, obj = self.get_url(url, {'username': 'abc', 'password': '123456'})
        self.assertEqual(code, 200)
        self.assertEqual(obj['status'], '200')
        self.assertEqual(obj['message'], 'success')


if __name__ == '__main__':
    unittest.main()
