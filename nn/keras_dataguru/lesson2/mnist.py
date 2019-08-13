import keras
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

(x_train,y_train),(x_test,y_test)=mnist.load_data()
print('x_shape:',x_train.shape) #(60000,28,28)
print('y_shape:',y_train.shape) #(60000,)

x_train = x_train.reshape(x_train.shape[0],-1)/255.0
x_test = x_test.reshape(x_test.shape[0],-1)/255.0

y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

model=Sequential([Dense(units=10,input_dim=784,bias_initializer='one',activation='softmax')])

sgd=SGD(lr=0.2)

model.compile(optimizer=sgd,loss='mse',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=10)

loss,accuracy=model.evaluate(x_test,y_test)

print('\ntest loss:',loss)
print('accuracy:',accuracy)