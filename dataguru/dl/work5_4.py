import numpy as np

a=np.random.random((50,1))
b=np.add.reduce(a)
c=np.sum(a)
d=np.add.reduce(a.flatten())
print(b)
print(c)
print(d)