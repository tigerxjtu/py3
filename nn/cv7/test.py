import math

def calc(a,b,c,x):
    return a*x**2+b*x+c

inputs=[-0.7192236, -2.7807765]
for i in inputs:
    print(calc(2,7,4,i))