#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: my_mail.py
@time: 2018-11-19
'''

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.application import MIMEApplication
import mimetypes
from pytest.testdata.getpath import GetConfigPath
import configparser
import os

class MyMail():

    def __init__(self):
        config = configparser.ConfigParser()
        config.read(GetConfigPath('mail.conf'))
        self.smtp=smtplib.SMTP()
        # self.smtp.set_debuglevel(1)

        conf=config['SMTP']
        self.login_user=conf['login_user']
        self.login_pwd=conf['login_pwd']
        self.from_addr=conf['from_addr']
        self.to_addrs = conf['to_addrs']
        self.host = conf['host']
        self.port = conf['port']


    # 连接到服务器
    def connect(self):
        # self.smtp.set_debuglevel(True)
        # self.smtp.starttls()
        print(self.host, self.port)
        print(self.host.find(':') , self.host.rfind(':'),type(self.host))
        self.smtp.connect(self.host, self.port)

    # 登陆邮件服务器
    def login(self):
        self.smtp.login(self.login_user, self.login_pwd)

    # 发送邮件
    def send_mail(self, mail_subject, mail_content, attachment_path_set):
        # 构造MIMEMultipart对象做为根容器
        msg = MIMEMultipart()
        msg['From'] = self.from_addr
        # msg['To'] = self.to_addrs
        msg['To'] = ','.join(eval(self.to_addrs))
        # 注意，这里的msg['To']只能为逗号分隔的字符串，形如 'sdxx@163.com', 'xdflda@126.com'
        msg['Subject'] = mail_subject
        # 添加邮件内容
        content = MIMEText(mail_content, "html", _charset='gbk')

        msg.attach(content)
        for attachment_path in attachment_path_set:
            if os.path.isfile(attachment_path):
                type, coding=mimetypes.guess_type(attachment_path)
                if type is None:
                    type='application/octet-stream'

                major_type,minor_type = type.split('/',1)
                with open(attachment_path,'rb') as file:
                    if major_type == 'text':
                        attachment=MIMEText(file.read(),_subtype=minor_type,_charset='gbk')
                    elif major_type == 'image':
                        attachment = MIMEImage(file.read(), _subtype=minor_type)
                    elif major_type == 'application':
                        attachment = MIMEApplication(file.read(), _subtype=minor_type)
                    elif major_type == 'audio':
                        attachment = MIMEAudio(file.read(), _subtype=minor_type)

                attachment_name=os.path.basename(attachment_path)
                attachment.add_header('Content-Disposition','attachment',filename=('gbk','',attachment_name))

                msg.attach(attachment)

        full_text = msg.as_string()
        self.smtp.sendmail(self.from_addr,eval(self.to_addrs), full_text)


    #退出
    def quit(self):
        self.smtp.quit()



if __name__=="__main__":
    mymail = MyMail()
    mymail.connect()
    mymail.login()
    mail_content='测试邮件'
    mail_title='测试'
    mymail.send_mail(mail_title,mail_content,[])
    mymail.quit()
