# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:08:39 2016

@author: Administrator
"""
from bs4 import BeautifulSoup
import re
import urllib.parse

class HtmlParser(object):
    
    def parse(self,page_url,html_cont):
        if page_url is None or html_cont is None:
            return
            
        soup=BeautifulSoup(html_cont,'html.parser',from_encoding='utf-8')
        node = soup.find('ul',id='thread_list')
        links = node.findAll('a', href=re.compile(r"/p/.+"))
        new_urls=set()
        new_data=[]
        for link in links:
            new_url = link['href']
            new_full_url = urllib.parse.urljoin(page_url, new_url)
            new_urls.add(new_full_url)
            res_data={'title':link.get_text(),'url':new_full_url}
            new_data.append(res_data)
        return new_urls,new_data