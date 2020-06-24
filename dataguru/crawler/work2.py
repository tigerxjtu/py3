import re

def extract_domain(url):
    regex = '(http|ftp|https)://(?P<domain>[\w\-_]+(\.[\w\-_]+)*)(/.*)*'
    result = re.match(regex,url.lower())
    if result:
        return result.group('domain')
    else:
        return ''

url = 'http://www.baidu.com/s?wd=%E7%88%AC%E8%99%AB&rsv_spt=1&rsv_iqid=0xea65e60300007d2a&issp=1&f=8&rsv_bp=0&rsv_idx=2&ie=utf-8&tn=02049043_62_pg&rsv_enter=1&rsv_sug3=5&rsv_sug1=3&rsv_sug7=100&rsv_sug2=0&inputT=1169&rsv_sug4=2873&rsv_sug=1'
print(extract_domain(url))