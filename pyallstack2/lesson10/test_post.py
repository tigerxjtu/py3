import requests

url = 'http://127.0.0.1:8000/api/add_author/'

data ={
    'id':12,
    'name':'lemon',
    'sex':'F',
    'telphone':'1300000000',
    'email':'lemon@abc.com'
}

response = requests.post(url=url,data=data)
print(response.status_code)
print(response.encoding)
print(response.json())