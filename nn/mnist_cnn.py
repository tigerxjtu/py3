import keras
from keras.datasets import mnist
# from keras.datasets import mnist
from keras.models import Model
from keras import models
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import numpy as np

batch_size = 120
num_classes = 10
epochs = 10

img_rows, img_cols = 28, 28

(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print(x_train.shape)  # (60000, 28, 28)

x_train = x_train.reshape(-1,img_rows,img_cols,1)
x_test = x_test.reshape(-1,img_rows,img_cols,1)

input_shape = (img_rows,img_cols,1)
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

# my_model = models.Sequential()
# my_model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=input_shape,name='CNN1'))
# my_model.add(Conv2D(64,kernel_size=(3,3), activation='relu',name='CNN2'))
# my_model.add(MaxPooling2D(pool_size=(2,2),name='MaxPool'))
# my_model.add(Flatten(name='Flatten'))
# my_model.add(Dense(256, activation='relu'))
# my_model.add(Dropout(0.02))
# my_model.add(Dense(num_classes,activation='softmax'))
#
# my_model.load_weights(r'C:\projects\python\py3\nn\my_model_weights.h5')

# x_test_cnn=my_model.predict(x_test[:2])
# print(my_model.layers[3].name)
# print(my_model.layers[3].output.shape)
# print(my_model.layers[3].output)
# print(x_test_cnn.shape)

# layer_name = 'Flatten' #获取层的名称
# intermediate_layer_model = Model(inputs=my_model.input,outputs=my_model.get_layer(layer_name).output)#创建的新模型
# intermediate_output = intermediate_layer_model.predict(x_test_cnn)
#
# print(intermediate_output.shape)
# print(intermediate_output)

with open('intermediate_layer_model.json','r') as f:
    json_str=f.read()

model=keras.models.model_from_json(json_str)
model.load_weights(r'cnn_weights.h5')

x_test_cnn=model.predict(x_test[:2])
print(x_test_cnn.shape)
print(x_test_cnn)
