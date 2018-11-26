#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: log.py
@time: 2018-11-05
'''

import logging
from logging.handlers import RotatingFileHandler
import threading
from pytest.testdata.getpath import GetTestLogPath


class LogSignleton(object):
    def __init__(self):
        pass


    def __new__(cls):
        mutex = threading.Lock()
        mutex.acquire()  # 上锁，防止多线程下出问题
        if not hasattr(cls, 'instance'):
            cls.instance = super(LogSignleton, cls).__new__(cls)
        cls.instance.log_filename = GetTestLogPath()
        cls.instance.max_bytes_each = 51200
        cls.instance.backup_count = 10
        cls.instance.fmt = "|(asctime)s |(filename)s[line: |(lineno)d] |(levelname)s: |(message)s"
        cls.instance.log_level_in_console = 10
        cls.instance.log_level_in_logfile = 20
        cls.instance.logger_name = "test_logger"
        cls.instance.console_log_on = 1
        cls.instance.logfile_log_on = 1
        cls.instance.logger = logging.getLogger(cls.instance.logger_name)
        cls.instance.__config_logger()
        mutex.release()
        return cls.instance


    def get_logger(self):
        return self.logger


    def __config_logger(self):
        # 设置日志格式
        fmt = self.fmt.replace('|', '%')
        formatter = logging.Formatter(fmt)
        if self.console_log_on == 1:  # 如果开启控制台日志
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            self.logger.addHandler(console)
            self.logger.setLevel(self.log_level_in_console)
        if self.logfile_log_on == 1:  # 如果开启文件日志
            rt_file_handler = RotatingFileHandler(
                self.log_filename, maxBytes=self.max_bytes_each, backupCount=self.backup_count)
            rt_file_handler.setFormatter(formatter)
            self.logger.addHandler(rt_file_handler)
            self.logger.setLevel(self.log_level_in_logfile)

logsignleton = LogSignleton()
logger = logsignleton.get_logger()
