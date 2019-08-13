import requests
import csv

base_url='http://127.0.0.1:8000/accounts/login'

url = 'http://127.0.0.1:8000/api/login/'

# data ={
#     'username':'user1',
#     'password':'pass@1234',
# }

def login(data):
    login_session = requests.Session()
    base_response = login_session.get(base_url)
    csrf_token = base_response.cookies['csrftoken']
    # print(csrf_token)

    data['csrfmiddlewaretoken']=csrf_token

    header = {"Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.80 Safari/537.36",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9"
    }

    response = login_session.post(url=url,data=data,headers=header)
    # print(response.status_code)
    # print(response.encoding)
    print(response.text)
    try:
        json=response.json()
        if json['message']=='success':
            print('登录成功')
            return 1
        else:
            print('登录失败')
            return 0
    except Exception as e:
        print(e)
        return -1

def read_records():
    headers = []
    records = []
    with open(r'C:\projects\python\py3\pyallstack2\lesson10\data.csv', newline='') as csvfile:  # 此方法:当文件不用时会自动关闭文件
        csvReader = csv.reader(csvfile)
        for content in csvReader:
            if not headers:
                headers = content
                continue
            data = {}
            for i, key in enumerate(headers):
                data[key] = content[i]
            records.append(data)
    return records

if __name__ == '__main__':
    records=read_records()
    for record in records:
        login(record)