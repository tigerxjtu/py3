import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print(x_train.shape)

x_train = x_train/255.0
x_test=x_test/255.0

y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
print(input_shape)

model=Sequential()
# 第一个卷积层
# input_shape 输入平面
# filters 卷积核/滤波器个数
# kernel_size 卷积窗口大小
# strides 步长
# padding padding方式 same/valid
# activation 激活函数
model.add(Convolution2D(
    input_shape = input_shape,
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    activation = 'relu'
))
# 第一个池化层
# model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))

# 第二个卷积层
model.add(Convolution2D(64,5,activation = 'relu'))
# 第二个池化层
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))

model.add(Flatten())

# 第一个全连接层
model.add(Dense(1024,activation = 'relu'))

# Dropout
model.add(Dropout(0.2))

# 第二个全连接层
model.add(Dense(10,activation='softmax'))

# 定义优化器
adam = Adam(lr=1e-4)

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# 训练模型
model.fit(x_train,y_train,batch_size=64,epochs=10)

# 评估模型
loss,accuracy = model.evaluate(x_test,y_test)

print('test loss',loss)
print('test accuracy',accuracy)