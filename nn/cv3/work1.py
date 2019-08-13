import numpy as np
import math
import matplotlib.pyplot as plt


p1=(0,2)
p2=(1,3)
p3=(3,5)

pts=[p1,p2,p3]
styles=['r*','g+','b-']
thet=np.arange(0,math.pi,0.05*math.pi)
index=0
plt.figure(figsize=(16,9))
for p in pts:
    x,y=p
    rous=[x*math.cos(the)+y*math.sin(the) for the in thet]
    plt.plot(thet,rous,styles[index])
    index += 1
plt.legend(loc='upper left',labels=pts)
plt.show()
