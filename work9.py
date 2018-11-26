import os

print('before patch:')
print(os.listdir('.'))
__origin_listdir__=os.listdir

def new_listdir(*args, **kwargs):
    for item in __origin_listdir__(*args, **kwargs):
        yield item

os.listdir=new_listdir
print('after patch:')
print(os.listdir('.'))
print([i for i in os.listdir('.')])
