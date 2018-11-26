#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: mysql_caozuo.py
@time: 2018-11-12
'''

host = "127.0.0.1"
user = "root"
password = "pass"
db = "polls"

import pymysql.cursors
class MySQLOperating():
    def __init__(self):
        try:
            # Connect to the database
            self.connection = pymysql.connect(host=host,
                user=user,
                password=password,
                db=db,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
                )
        except pymysql.err.OperationalError as e:
            print("Mysql Error %d: %s" % (e.args[0], e.args[1]))
            raise e

    def clear(self, table_name):
        real_sql = "delete from " + table_name + ";"

        with self.connection.cursor() as cursor:
            cursor.execute("SET FOREIGN_KEY_CHECKS=0;")
            cursor.execute(real_sql)
        self.connection.commit()

    def insert(self, table_name, data):
        for key in data:
            data[key] = "'" + str(data[key]) + "'"
        key = ','.join(data.keys())
        value = ','.join(data.values())
        real_sql = "INSERT INTO " + table_name + " (" + key + ") VALUES (" + value + ")"
        print(real_sql)
        with self.connection.cursor() as cursor:
            cursor.execute(real_sql)
        self.connection.commit()

    def close(self):
        self.connection.close()