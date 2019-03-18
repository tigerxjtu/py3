# coding: utf-8

'''

涉及内容:信用卡客户流失预警模型-CRISP_DM建模流程、数据清洗、变量压缩、模型开发与评估

1、背景介绍：

随着信用卡市场的成熟，人均持卡量不断增加，加上第三方支付牌照的持续发放，人们可选择的支付手段不断丰富，信用卡客户流失（销卡）呈现常态化。C银行在国内信用卡市场中处于领先地位，管理层非常重视客户生命周期管理并取得了良好的回报，为进一步完善对客户流失及挽留环节的管理，管理层要求建立大数据模型，基于对客户销卡决心和预期价值的准确预测，制定差异化挽留策略，实现收益与成本的最佳平衡。具体来说，当客户打进电话提出销卡时，将客户的销卡决心、预期价值以及相应的应对策略，展示在客服人员的工作指导窗口上，在客户挽留环节改进客户体验，加强对潜在高价值客户的挽留力度。。

本次作业根据提供的数据（“CSR_CHURN_Samp.csv”，引用自陈春宝等出版的《SAS金融数据挖掘与建模》）信用卡客户流失预警模型。

2、本案例涉及的部分变量说明如下：

STA_DTE 数据提取时间

Evt_Flg 是否流失(作为被预测变量Y)

Value 客户的价值,本次作业中不需要参与建模

Age	年龄

Gen	性别，1=男

Buy_Type	近一个月主要的购物类型

R3m_Avg_Cns_Amt	近3个月月均消费金额

R6m_Avg_Rdm_Pts	近6个月月均兑换积分

R12m_Avg_Cns_Cnt	近12个月月均消费次数

R6m_Cls_Nbr	近半年还款拖欠次数

Ilt_Bal_Amt	当前分期未还余额

Lmth_Fst_Ilt	累计分期产品办理次数

Lmth_Fst_Int	累计小额信贷申请次数

Csr_Dur	累计持卡时长

R6m_Call_Nbr	近半年投诉次数

Total_Call_Nbr	累计投诉次数

Net_Cns_Cnt	累计网上交易次数

Ovs_Cns_Amt	累计境外交易次数

其他略：学习到这个阶段，已经可以适应不需要知道变量含义，凭借数据分析工序建立分类模型的状态。
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#需要把woe所在的目录"D:\Python_book\19Case\19_2Donations"设置到python工作目录，
#设置方式为Tools->PYTHONPATH manager

import os
os.chdir(r"C:\projects\python\py3\datasci")
from .woe import WoE # 从本地导入

# 创建一个列表，用来保存所有的建模数据清洗的相关信息
DATA_CLEAN = []

model_data = pd.read_excel("CSR_CHURN_Samp.xlsx")
model_data.head()