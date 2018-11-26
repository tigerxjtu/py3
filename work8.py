# -*-coding:utf-8-*-

import sys
import importlib

def has_attr(module, attr):
    mod = importlib.import_module(str(module))
    return hasattr(mod,attr)

if len(sys.argv)!=3:
    print('usage: python work8.py module attribute')
    exit(1)
# print(sys.argv[1])
# print(sys.argv[2])
print(has_attr(sys.argv[1], sys.argv[2]))
