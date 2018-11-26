#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: init_data.py
@time: 2018-11-12
'''
from pytest.lesson9.mysql_caozuo import MySQLOperating

table_poll_question = "polls_question"
datas_poll_question =[ {'id': 1, 'question_text': '你喜欢的游戏是什么?'},{'id': 2, 'question_text': '你喜欢去哪里旅游?'}]
table_poll_choice = "polls_choice"
datas_poll_choice =[{'id': 1, 'choice_text': '生化危机', 'votes': 0, 'question_id': 1},
                    {'id': 2, 'choice_text': 'GTA5', 'votes': 0, 'question_id': 1},
                    {'id': 3, 'choice_text': '新加坡', 'votes': 0, 'question_id': 2},
                    {'id': 4, 'choice_text': '泰国', 'votes': 0, 'question_id': 2},
                    ]
table_poll_user = "polls_user"
datas_poll_user=[{'id': 1,'user_name': 'abc', 'password': '123456'}]

def insert_data(table, datas):
    db = MySQLOperating()
    db.clear(table)
    for data in datas:
        db.insert(table, data)
    db.close()


def init_data():
    insert_data(table_poll_question, datas_poll_question)
    insert_data(table_poll_choice, datas_poll_choice)
    insert_data(table_poll_user, datas_poll_user)

if __name__ == '__main__':
    init_data()