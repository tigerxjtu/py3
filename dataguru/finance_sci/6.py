# -*- coding: utf-8 -*-

import numpy as np

fixIR=[]
fixYE=[]
fixBJ=[]
# 等额还款
def fixpayment(MP,Num,B,rate):
    # MP 月还款额
    # Num 期数
    # B 贷款本金
    # rate 贷款利率
    # 初始化相关变量
    # 初始化行向量，用来存储每次循环的值
    IR = [0] * Num # 月利息
    YE = [0] * Num  # 贷款余额
    BJ = [0] * Num  # 每月偿还本金
    # 第一个月初贷款余额等于本金
    YE[0] = B
    for i in range(Num):
        IR[i] = Rate * YE[i]
        BJ[i] = MP - IR[i]
        # 不是最后一次还款
        if i < Num - 1:
            YE[i+1] = YE[i] - BJ[i] #第i+1期本金等于第i期本金 - 第i期归还本金
    global fixIR
    global fixYE
    global fixBJ
    
    fixIR=IR
    fixYE=YE
    fixBJ=BJ
    
    return B - sum(BJ) #剩余本金

#设置相关参数
Num =240
B = 1000000
Rate = 0.06/12

#找到MP(每月应该还款多少)
from scipy.optimize import fsolve
# 初始值5000
Mpo = 10000
MP = fsolve(lambda MP: fixpayment(MP,Num,B,Rate), Mpo)
print ('每月贷款应还款为%.2f元' % MP)
fixpayment(MP,Num,B,Rate)
fixMP=[MP]*Num


##用公式
#r =0.06/12
#MP = 1000000* (r*(1+r)**240)/ ((1+r)**240 -1 )
#print ('每月贷款应还款为%.2f元' % MP)

varIR=[]
varYE=[]
varMP=[]
# # # # # # 等额本金还款
def varPayment(Num,B,Rate):
    MB = B/Num #每月本金
    # 初始化相关变量
    IR = [0]*Num #每月利息
    YE = [0]*Num #每月贷款余额
    MP = [0]*Num  #每月还款额
    YE[0] = B
    for i in range(0,Num):
        IR[i] = YE[i] * Rate
        MP[i] = MB + IR[i]
        if i < Num - 1:
            YE[i + 1] = YE[i] - MB
    global varIR
    global varYE
    global varMP
    
    varIR=IR
    varYE=YE
    varMP=MP
    
    return MP
## 设置参数
#Rate = 0.05/12
#Num =120
#B =500000
MP = varPayment(Num,B,Rate)
print ('每月贷款应还款为%s \n' % MP)
varBJ=[item[0]-item[1] for item in zip(varMP,varIR)]



#绘制两种还款计划图
import matplotlib.pyplot as plt
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(varMP,label ='等额本金')
plt.plot(fixMP,label='等额还款')
plt.legend(loc='upper right')
plt.show()

#每月还款比较
plt.plot(varMP,label ='等额本金-还款')
plt.plot(fixMP,label='等额还款-还款')
plt.legend(loc='upper right')
plt.show()

#本金比较
plt.plot(varBJ,label ='等额本金-本金')
plt.plot(fixBJ,label='等额还款-本金')
plt.legend(loc='upper right')
plt.show()

#利息比较
plt.plot(varIR,label ='等额本金-利息')
plt.plot(fixIR,label='等额还款-利息')
plt.legend(loc='upper right')
plt.show()

#剩余本金比较
plt.plot(varYE,label ='等额本金-剩余本金')
plt.plot(fixYE,label='等额还款-剩余本金')
plt.legend(loc='upper right')
plt.show()

print('等额还款方式总还款：%.2f元'%sum(fixMP))
print('等额本金方式总还款：%.2f元'%sum(varMP))

print('总还款差值:%.2f元'%(sum(fixMP)-sum(varMP)))