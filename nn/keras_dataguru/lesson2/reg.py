import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential  #顺序模型
from keras.layers import Dense #全连接层


x_data=np.linspace(-0.5,0.5,200)
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

plt.scatter(x_data,y_data)
plt.show()

model=Sequential()
# 1-1
# model.add(Dense(units=1,input_dim=1)) #一个输出，一个输入

#1-10-1
model.add(Dense(units=10,input_dim=1,activation='tanh'))
# model.add(keras.layers.Activation('tanh'))
model.add(Dense(units=1,activation='tanh'))
# model.add(keras.layers.Activation('tanh'))
model.compile(optimizer=keras.optimizers.SGD(lr=0.3),loss=keras.losses.mse)

for step in range(3000):
    cost=model.train_on_batch(x_data,y_data)
    if step % 500 ==0:
        print('cost:',cost)

W,b=model.layers[0].get_weights()
print('W:',W,'b:',b)

y_pred=model.predict(x_data)
plt.scatter(x_data,y_data)
plt.plot(x_data,y_pred,'r--',lw=3)
plt.show()