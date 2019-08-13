import requests

def test_user_auth_null(url,args):
    response = requests.get(url,args)
    result = response.json()
    print(result)

def test_user_auth_error(url,args):
    user=('user1','pass@1234')
    response = requests.get(url,auth=user,params=args)
    result = response.json()
    print(result)

def test_user_auth_ok(url,args):
    user=('user1','pass@1234')
    response = requests.get(url,auth=user,params=args)
    result = response.json()
    print(result)


if __name__ == '__main__':
    url = 'http://127.0.0.1:8000/api/get_authors_sec'
    author = dict(id='3',name='user2')
    test_user_auth_null(url,author)
    test_user_auth_error(url,author)
    test_user_auth_ok(url,{'id':'','name':''})
    test_user_auth_ok(url, {'id': '3'})
    test_user_auth_ok(url, {'name': 'user1'})
    test_user_auth_ok(url, {'id': '3', 'name': 'user2'})
