# -*-coding:utf-8-*-

def except_caught(err=Exception):
    def decorator(func):
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except err as e:
                print("Got exception! ", e)
        return new_func
    return decorator

@except_caught()
def divide(a,b):
    return a/b

print(divide(10,2))
print(divide(10,0))
