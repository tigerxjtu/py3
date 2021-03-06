# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:07:51 2016

@author: Administrator
"""
from dataguru.crawler.lesson6 import url_manager,html_downloader,html_parser,html_outputer


class SpiderMain(object):
    def __init__(self):
        self.urls=url_manager.UrlManager()
        self.downloader=html_downloader.HtmlDownloader()
        self.parser=html_parser.HtmlParser()
        self.outputer=html_outputer.HtmlOutputer()
        
    def craw(self,root_rul):
        count = 1
        self.urls.add_new_url(root_url)
        while self.urls.has_new_url():
            try:
                new_url=self.urls.get_new_url()
                print('craw %d:%s' %(count,new_url))
            except:
                print(1)
            try:               
                html_cont=self.downloader.download(new_url)
            except: 
                print(2)
            try: 
                new_urls,new_data=self.parser.parse(new_url,html_cont)
            except:
                print(3)
            try:
                self.urls.add_new_urls(new_urls)
            except:
                print(4)
            try:
                self.outputer.output_html(new_data)
            except:
                print(5)

            count=count+1
            if count == 100:
                break
#            except:
#                print 'craw failed'

            


if __name__=="__main__":
    # root_url='https://baike.baidu.com/item/%E5%A4%A7%E5%A3%81%E8%99%8E/292642?fromtitle=%E8%9B%A4%E8%9A%A7&fromid=1604'
    root_url='https://baike.baidu.com/item/%E7%BB%9F%E8%AE%A1%E5%AD%A6/1175'
    obj_spider = SpiderMain()
    obj_spider.craw(root_url)
    