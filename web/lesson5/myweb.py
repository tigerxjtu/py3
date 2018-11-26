#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: myweb.py
@time: 2018-11-14
'''

import web
import json

render = web.template.render('template/')

urls = (
    '/showfile/(.+)','ShowFile',
    '/index','index',
    '/search/(.+)','search',
    '/(.*)', 'hello'
)
app = web.application(urls, globals())

keywords=['John','Joe','Jack','Kite','Alice','Bob']

class ShowFile:
    def GET(self,filename):
        web.header('content-type', 'text/html')
        with open('./template/%s'%filename,'r',encoding='utf-8') as f:
            return f.read()

class index:
    def GET(self):
        # return 'Hello world'
        return render.index()

class hello:
    def GET(self, name):
        if not name:
            name = 'World'
        return 'Hello, ' + name + '!'

class search:
    def GET(self,key):
        result=filter(lambda x:x.startswith(key),keywords)
        return json.dumps(list(result))



if __name__ == "__main__":
    app.run()