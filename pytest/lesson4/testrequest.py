#!/usr/bin/python3
# encoding:utf-8

'''
@author: liyin
@file: testrequest.py
@time: 2018-10-06
'''

import json
import requests
from pytest.lesson8.log import logger

# 添加一个数组，用来装测试结果
hlist = []
# 公共的头文件设置
header = {
    'content-type': "application/json;charset=UTF-8"
}


def TestPostRequest(hurl, hdata, htestcassid, htestcassname, htesthope, fanhuitesthope,headers=header):
    hr = requests.post(hurl, data=hdata, headers=headers)
    # print(hr.text)
    hjson = json.loads(hr.text)  # 获取并处理返回的json数据
    herror = "error"
    if herror in hjson:
        hstatus = str(hjson["status"])
        if hstatus == htesthope and fanhuitesthope in str(hjson):
            hhhdata = {"t_id": htestcassid,
                       "t_name": htestcassname,
                       "t_method": "post",
                       "t_url": hurl,
                       "t_param": "测试数据:" + str(hdata),
                       "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                       "t_actual": "status:" + hstatus + ";msg:" + str(hjson),
                       "t_result": "通过"}
            hlist.append(hhhdata)  # 把测试结果添加到数组里面
            logger.info(htestcassid)
            logger.info("通过")
            logger.info("返回结果："+str(hjson))
        else:
            hhhdata = {"t_id": htestcassid,
                       "t_name": htestcassname,
                       "t_method": "post",
                       "t_url": hurl,
                       "t_param": "测试数据:" + str(hdata),
                       "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                       "t_actual": str(hjson),
                       "t_result": "失败"}
            hlist.append(hhhdata)
            logger.error(htestcassid)
            logger.error("失败")
            logger.error("返回结果：" + str(hjson))
    else:
        if "'status_code': 500" in str(hjson) or "'status_code': 404" in str(hjson):
            hstatus = str(hjson["status_code"])
            hhhdata = {"t_id": htestcassid,
                       "t_name": htestcassname,
                       "t_method": "get",
                       "t_url": hurl,
                       "t_param": "测试数据:" + str(hdata),
                       "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                       "t_actual": "status:" + hstatus + ";msg:" + str(hjson),
                       "t_result": "失败"}
            hlist.append(hhhdata)
            logger.error(htestcassid)
            logger.error("失败")
            logger.error("返回结果：" + str(hjson))
        else:
            hcode = str(hjson['status'])
            if hcode == htesthope and fanhuitesthope in str(hjson):
                hhhdata = {"t_id": htestcassid,
                           "t_name": htestcassname,
                           "t_method": "get",
                           "t_url": hurl,
                           "t_param": "测试数据:" + str(hdata),
                           "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                           "t_actual": "status:" + hcode + ";data:" + str(hjson),
                           "t_result": "通过"}
                hlist.append(hhhdata)  # 把测试结果添加到数组里面
                logger.info(htestcassid)
                logger.info("通过")
                logger.info("返回结果：" + str(hjson))
            else:
                hhhdata = {"t_id": htestcassid,
                           "t_name": htestcassname,
                           "t_method": "get",
                           "t_url": hurl,
                           "t_param": "测试数据:" + str(hdata),
                           "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                           "t_actual": "status:" + hcode + ";msg:" + str(hjson),
                           "t_result": "失败"}
                hlist.append(hhhdata)
                logger.error(htestcassid)
                logger.error("失败")
                logger.error("返回结果：" + str(hjson))
    return hlist


def TestGetRequest(hurl, hdata, htestcassid, htestcassname, htesthope, fanhuitesthope,headers=header, status_key='status'):
    hr = requests.get(hurl, params=hdata, headers=headers)
    # print(hr.text)
    hjson = json.loads(hr.text)  # 获取并处理返回的json数据
    herror = "error"
    if herror in hjson:
        hstatus = str(hjson[status_key])
        if hstatus == htesthope and fanhuitesthope in str(hjson):
            hhhdata = {"t_id": htestcassid,
                       "t_name": htestcassname,
                       "t_method": "get",
                       "t_url": hurl,
                       "t_param": "测试数据:" + str(hdata),
                       "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                       "t_actual": "status:" + hstatus + ";msg:" + str(hjson),
                       "t_result": "通过"}
            hlist.append(hhhdata)  # 把测试结果添加到数组里面
            logger.info(htestcassid)
            logger.info("通过")
            logger.info("返回结果：" + str(hjson))
        else:
            hhhdata = {"t_id": htestcassid,
                       "t_name": htestcassname,
                       "t_method": "get",
                       "t_url": hurl,
                       "t_param": "测试数据:" + str(hdata),
                       "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                       "t_actual": str(hjson),
                       "t_result": "失败"}
            hlist.append(hhhdata)
            logger.error(htestcassid)
            logger.error("失败")
            logger.error("返回结果：" + str(hjson))
    else:
        if "'status_code': 500" in str(hjson) or "'status_code': 404" in str(hjson):
            hstatus = str(hjson["status_code"])
            hhhdata = {"t_id": htestcassid,
                       "t_name": htestcassname,
                       "t_method": "get",
                       "t_url": hurl,
                       "t_param": "测试数据:" + str(hdata),
                       "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                       "t_actual": "status:" + hstatus + ";msg:" + str(hjson),
                       "t_result": "失败"}
            hlist.append(hhhdata)
            logger.error(htestcassid)
            logger.error("失败")
            logger.error("返回结果：" + str(hjson))
        else:
            hcode = str(hjson[status_key])
            if hcode == htesthope and fanhuitesthope in str(hjson):
                hhhdata = {"t_id": htestcassid,
                           "t_name": htestcassname,
                           "t_method": "get",
                           "t_url": hurl,
                           "t_param": "测试数据:" + str(hdata),
                           "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                           "t_actual": "status:" + hcode + ";data:" + str(hjson),
                           "t_result": "通过"}
                hlist.append(hhhdata)  # 把测试结果添加到数组里面
                logger.info(htestcassid)
                logger.info("通过")
                logger.info("返回结果：" + str(hjson))
            else:
                hhhdata = {"t_id": htestcassid,
                           "t_name": htestcassname,
                           "t_method": "get",
                           "t_url": hurl,
                           "t_param": "测试数据:" + str(hdata),
                           "t_hope": "status:" + str(htesthope) + " 包含：" + str(fanhuitesthope),
                           "t_actual": "status:" + str(hcode) + ";msg:" + str(hjson),
                           "t_result": "失败"}
                hlist.append(hhhdata)
                logger.error(htestcassid)
                logger.error("失败")
                logger.error("返回结果：" + str(hjson))
    return hlist


def TestDeleteRequest(hurl, hdata, htestcassid, htestcassname, htesthope, fanhuitesthope,headers=header):
    hr = requests.delete(hurl, params=hdata, headers=headers)
    hjson = json.loads(hr.text)  # 获取并处理返回的json数据
    herror = "error"
    if herror in hjson:
        hstatus = str(hjson["status"])
        if hstatus == htesthope and fanhuitesthope in str(hjson):
            hhhdata = {"t_id": htestcassid,
                       "t_name": htestcassname,
                       "t_method": "post",
                       "t_url": hurl,
                       "t_param": "测试数据:" + str(hdata),
                       "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                       "t_actual": "status:" + hstatus + ";msg:" + str(hjson),
                       "t_result": "通过"}
            hlist.append(hhhdata)  # 把测试结果添加到数组里面
            logger.info(htestcassid)
            logger.info("通过")
            logger.info("返回结果：" + str(hjson))
        else:
            hhhdata = {"t_id": htestcassid,
                       "t_name": htestcassname,
                       "t_method": "post",
                       "t_url": hurl,
                       "t_param": "测试数据:" + str(hdata),
                       "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                       "t_actual": str(hjson),
                       "t_result": "失败"}
            hlist.append(hhhdata)
            logger.error(htestcassid)
            logger.error("失败")
            logger.error("返回结果：" + str(hjson))
    else:
        if "'status_code': 500" in str(hjson) or "'status_code': 404" in str(hjson):
            hstatus = str(hjson["status_code"])
            hhhdata = {"t_id": htestcassid,
                       "t_name": htestcassname,
                       "t_method": "get",
                       "t_url": hurl,
                       "t_param": "测试数据:" + str(hdata),
                       "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                       "t_actual": "status:" + hstatus + ";msg:" + str(hjson),
                       "t_result": "失败"}
            hlist.append(hhhdata)
            logger.error(htestcassid)
            logger.error("失败")
            logger.error("返回结果：" + str(hjson))
        else:
            hcode = str(hjson['status'])
            if hcode == htesthope and fanhuitesthope in str(hjson):
                hhhdata = {"t_id": htestcassid,
                           "t_name": htestcassname,
                           "t_method": "get",
                           "t_url": hurl,
                           "t_param": "测试数据:" + str(hdata),
                           "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                           "t_actual": "status:" + hcode + ";data:" + str(hjson),
                           "t_result": "通过"}
                hlist.append(hhhdata)  # 把测试结果添加到数组里面
                logger.info(htestcassid)
                logger.info("通过")
                logger.info("返回结果：" + str(hjson))
            else:
                hhhdata = {"t_id": htestcassid,
                           "t_name": htestcassname,
                           "t_method": "get",
                           "t_url": hurl,
                           "t_param": "测试数据:" + str(hdata),
                           "t_hope": "status:" + str(htesthope) + " 包含：" + fanhuitesthope,
                           "t_actual": "status:" + hcode + ";msg:" + str(hjson),
                           "t_result": "失败"}
                hlist.append(hhhdata)
                logger.error(htestcassid)
                logger.error("失败")
                logger.error("返回结果：" + str(hjson))
    return hlist
