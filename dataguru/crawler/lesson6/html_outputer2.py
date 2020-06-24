# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:09:01 2016

@author: Administrator
"""

class HtmlOutputer(object):

    def output_html(self,datas):
            if datas is None:
                return
            for data in datas:
                print('title:%s, url:%s'%(data['title'],data['url']))
        
        
        