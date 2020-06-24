import numpy as np

a=np.random.random((10,10))
b=np.sort(a,axis=0)
result = b[-1:-4:-1,:]
print(a,b)
print('result:----------------------------------------')
print(result)