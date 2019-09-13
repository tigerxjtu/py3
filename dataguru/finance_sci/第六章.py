# 第六章代码
import numpy as np
#==============================================================================
# 6.1 货币时间价值
#==============================================================================
# 6.1.1 单利终值和现值
FV = 10000;T=3;r=0.05
PV  =10000/(1+0.05*3)
print ('现在应该往银行存入%.2f元' % PV)

# 6.1.2 复利终值和现值
FV = 10000;T=3;r=0.05
PV  =10000/(1+0.05)**3
print ('现在应该往银行存入%.2f元' % PV)
PV  =10000/(1+0.05/2)**6
print ('现在应该往银行存入%.2f元' % PV)

# 6.1.3 连续复利
#年化利率为5%，按照每个月复利一次
PV  =10000* np.exp(-3*r)
print ('现在应该往银行存入%.2f元' % PV)

# 调整复利计息频数
# 计算实际每年收益率
R_continous =[]
R = 0.05
for i in range(1,366):
    r = (1+R/i)**i
    R_continous.append(r)
limit = np.ones(len(R_continous))* np.exp(R)
import matplotlib.pyplot as plt
plt.plot(R_continous,label='continous')
plt.plot(limit,label = 'limit' )
plt.legend(loc='upper right')
#plt.title('')
plt.show()

#==============================================================================
# 6.2 固定现金流计算
#==============================================================================
# 固定现金流计算PV
FaceValue = 1000
Payment =1000*0.05 # 债券收益(每期)
Rate = 0.06 #贴现率
final_payment = 1000 #到期还本,也可以理解为FV
Num = 10
Due = 0 #现金流计息方式(0为周期末付息，1为周期内付息)
Presentvalue = abs(np.pv(Rate,Num,Payment,final_payment,Due))
print ('当前现值为%.2f元' % Presentvalue)
# 固定现金流计算FV
FutureValue = abs(np.fv(Rate,Num,Payment,final_payment,Due))
print ('当前终值为%.2f元' % FutureValue)

#用自己编写函数
def PV(face_value, payment, num,r):
    Pv_v = 0  # 初始值(是每期收益贴现后的现值
    for i in range(1,num+1):
        pv = 50/(1+0.06)**i
        Pv_v  = Pv_v + pv
    face_value_p = face_value/(1+r)**num #最后一次现金流贴现
    return Pv_v + face_value_p

Presentvalue = PV(1000, 50, 10,0.06)
print ('当前现值为%.2f元' % Presentvalue)


#用自己编写函数
def FV(face_value, payment, num,r):
    Fv_v = 0
    for i in np.arange(0, 10):
        fv = 50*(1+r)**i
        Fv_v  = Fv_v + fv
    face_value_f = face_value*(1+r)**num
    return Fv_v + face_value_f

Future_value = FV(1000, 50, 10,0.06)
print ('终值为%.2f元' % Future_value )

#==============================================================================
# 6.3 变化现金流计算
#==============================================================================
# NPV
import scipy as sp
cashflows=[-10000,3000,3500,7000,8000,2000]
sp.npv(0.112,cashflows)
if sp.npv(0.112,cashflows) > 0:
    print ('投资可以接受')
else:
    print('投资不可行')

# IRR
# 计算IRR
cashflows = [-6000,2500,1500,3000,1000,2000]
IRR = np.irr(cashflows)
print('内部收益率为: %.4f%%' % (100*IRR))

#使用函数
def IRR_rate(initial_invest,cash_flow,R):
    cash_total = 0
    for i in range(0,len(cash_flow)):
        cash_in = cash_flow[i]/(1+R)**(i+1)
        cash_total = cash_total +cash_in
    npv = initial_invest + cash_total
    return npv
# 用数值方法求解
from scipy.optimize import fsolve
initial_invest = -8000
cash_flow = [2500,1500,3000,1000,2000]

irr = fsolve(lambda R: IRR_rate(initial_invest,cash_flow,R),0.08)
print('内部收益率为: %.4f%%' % (100*irr))

#==============================================================================
# 6.4 年金现金流计算
#==============================================================================
#计算年化利率
Num =240
payment =-3000
PV =500000
FV =0
due = 0 #期末还款
Rate = np.rate(Num,payment,PV,FV,due)
print('年化利率为: %.4f%%' % (100*Rate*12))
#计算周期

rate = 0.038862/12
payment =-4000
PV =500000
FV =0
Periods = np.nper(rate,payment,PV,FV,when='end')
print('周期为:%s' % Periods )

#==============================================================================
# 6.5 按揭贷款分析
#==============================================================================
# 等额还款
def Ajfixpayment(MP,Num,B,rate):
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
    return B - sum(BJ) #剩余本金

#设置相关参数
Num =120
B = 500000
Rate = 0.05/12
MP = 5000
F = Ajfixpayment(MP,Num,B,Rate)
print ('贷款余额还剩下: %.2f' % F)

#找到MP(每月应该还款多少)
from scipy.optimize import fsolve
# 初始值5000
Mpo = 5000
MP = fsolve(lambda MP: Ajfixpayment(MP,Num,B,Rate), Mpo)
print ('每月贷款应还款为%.2f元' % MP)

#用公式
r =0.05/12
MP = 500000* (r*(1+r)**120)/ ((1+r)**120 -1 )
print ('每月贷款应还款为%.2f元' % MP)

# # # # # # 等额本金还款
def AJvarPayment(Num,B,Rate):
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
    return MP
# 设置参数
Rate = 0.05/12
Num =120
B =500000
MP = AJvarPayment(Num,B,Rate)
print ('每月贷款应还款为%s \n' % MP)


#绘制两种还款计划图
import matplotlib.pyplot as plt
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
r= 0.05/12
MP_fix = np.ones(len(MP))*500000* (r*(1+r)**120)/ ((1+r)**120 -1 )
plt.plot(MP,label ='等额本金')
plt.plot(MP_fix,label='等额')
plt.legend(loc='upper right')
plt.show()

