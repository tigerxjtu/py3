from math import *

p=-1

w1=-1
b1=1

w2=-2
b2=1

n1=w1*p+b1
a1=1 / (1 + exp(-n1))

n2=w2*a1+b2
a2=n2

y=-1
loss=(y-a2)* (y-a2)

print(n1,a1,n2,a2)


d_loss_w1=2*sqrt(loss)*a2*w2*exp(-n1)*p/(1 + exp(-n1))**2
d_loss_w2=2*a2*a1*sqrt(loss)

alpha=0.1
w1=w1+alpha* d_loss_w1
w2=w2+alpha* d_loss_w2
