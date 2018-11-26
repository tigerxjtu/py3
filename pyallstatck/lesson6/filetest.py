#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: filetest.py
@time: 2018-10-30
'''

import os
import shutil

if not os.path.exists("test"):
    os.mkdir("test")

def createFile(path, content):
    with open(path,'w') as f:
        f.write(content)


createFile('test/test.txt','txt')
createFile('test/test.xml','xml')
createFile('test/test.excel','excel')

def readFile(path):
    with open(path,'r') as f:
        return f.read()

print(readFile('test/test.txt'))
print(readFile('test/test.xml'))
print(readFile('test/test.excel'))

def moveFile(src,dst):
    shutil.move(src,dst)

moveFile('test/test.txt','test/test.txt.new')

def renameFile(path,src,dst):
    srcFile=os.path.join(path,src)
    dstFile=os.path.join(path,dst)
    shutil.move(srcFile, dstFile)

renameFile('test','test.xml','test.xml.new')

def deleteFile(path):
    os.remove(path)

deleteFile('test/test.excel')
