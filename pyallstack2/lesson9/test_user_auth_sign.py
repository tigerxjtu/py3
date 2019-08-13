import requests
from pyallstack2.lesson9.crypto import aes_crypt
import hashlib

def build_args(args):
    key = 'key_token'
    aes = aes_crypt(key)
    id = args.get('id','')
    name = args.get('name','')
    if 'id' in args:
        eid = aes.encrypt(id)
        args['id']=eid
    sign_str = id + name + key
    sign_str_utf8 = sign_str.encode(encoding='utf-8')
    md5 = hashlib.md5()
    md5.update(sign_str_utf8)
    sign = md5.hexdigest()
    args['sign']=sign
    return args


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
    response = requests.get(url,auth=user,params=build_args(args))
    result = response.json()
    print(result)


if __name__ == '__main__':
    url = 'http://127.0.0.1:8000/api/get_authors_sec'
    url1 = 'http://127.0.0.1:8000/api/get_authors_sign'
    author = dict(id='2',name='user1')
    test_user_auth_null(url,author)
    test_user_auth_error(url,author)
    test_user_auth_ok(url1,{'id':'','name':''})
    test_user_auth_ok(url1, {'id': '2'})
    test_user_auth_ok(url1, {'name': 'user1'})
    test_user_auth_ok(url1, {'id': '3', 'name': 'user2'})
