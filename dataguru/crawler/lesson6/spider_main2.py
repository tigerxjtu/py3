# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:07:51 2016

@author: Administrator
"""
from dataguru.crawler.lesson6 import url_manager,html_downloader,html_parser2,html_outputer2


class SpiderMain2(object):
    def __init__(self):
        self.urls=url_manager.UrlManager()
        self.downloader=html_downloader.HtmlDownloader()
        self.parser=html_parser2.HtmlParser()
        self.outputer=html_outputer2.HtmlOutputer()
        
    def craw(self,root_rul):
        count = 1
        self.urls.add_new_url(root_url)
        # while self.urls.has_new_url():
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
        except Exception as e:
            print(e)
            print(3)
        try:
            self.urls.add_new_urls(new_urls)
        except:
            print(4)
        try:
            self.outputer.output_html(new_data)
        except:
            print(5)

            # count=count+1
            # if count == 100:
            #     break
#            except:
#                print 'craw failed'

            


if __name__=="__main__":
    # root_url='https://baike.baidu.com/item/%E5%A4%A7%E5%A3%81%E8%99%8E/292642?fromtitle=%E8%9B%A4%E8%9A%A7&fromid=1604'
    root_url='https://tieba.baidu.com/f?kw=%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB&traceid='
    obj_spider = SpiderMain2()
    obj_spider.craw(root_url)
    