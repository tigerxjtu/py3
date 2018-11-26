# -*-coding:utf-8-*-
from gevent import monkey; monkey.patch_all()
import gevent
import requests
import os

headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}

def makedir(dirname):
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    folder=os.path.join(parent_dir,dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def downloadImg(url):
    if url:
        response=requests.get(url,headers=headers)
        filename=url.split('/')[-1]
        path=os.path.join(makedir('images'),filename)
        f=open(path,'wb')
        f.write(response.content)
        f.close()

#
# def f(url):
#     print('GET: %s' % url)
#     resp = requests.get(url)
#     data = resp.text
#     print('%d bytes received from %s.' % (len(data), url))

gevent.joinall([
        gevent.spawn(downloadImg, 'http://pub.lqtedu.com/upload/login/2018-06-05/2b5b80d9-7dff-4dbe-8394-70c64570032b.jpg'),
        gevent.spawn(downloadImg, 'http://n.sinaimg.cn/translate/w500h320/20171129/YIkR-fypceiq7072067.jpg'),
])