import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

input=[[0,1],[1,1],[1,1],[0,0]]
padding=[[0,0]]*3
inputs=padding+input
time_steps=4
input_size=2
samples=4
x=np.zeros((samples,time_steps,input_size))
for i in range(samples):
    x[i,:,:]=np.array(inputs[i:i+time_steps])
# (4,4,2) (sample num，time_steps, input_size)
# x=np.array(x).reshape(-1,time_steps,input_size)
# print(x[0])

y=[1,0,1,1]
y=np.array(y).reshape(-1,1)
y=np_utils.to_categorical(y)
print(y)

# input_shape = (time_steps,input_size)
model = Sequential()
model.add(LSTM(
    units = 16, # 输出
    input_shape = (time_steps,input_size), #输入
))

model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,epochs=10,batch_size=4)

# print(model.predict(x))

# # 评估模型
# loss,accuracy = model.evaluate(x,y)
#
# print('test loss',loss)
# print('test accuracy',accuracy)