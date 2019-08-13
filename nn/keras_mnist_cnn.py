
import keras
from keras.datasets import mnist
from keras import models
from keras.models import Model
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
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

# print(y_train)  #onehot

my_model = models.Sequential()
my_model.add(Conv2D(32,kernel_size=(5,5), activation='relu', input_shape=input_shape,name='CNN1'))
my_model.add(MaxPooling2D(pool_size=(2,2),name='MaxPool1'))
my_model.add(Conv2D(64,kernel_size=(5,5), activation='relu',name='CNN2'))
my_model.add(MaxPooling2D(pool_size=(2,2),name='MaxPool2'))
my_model.add(Flatten(name='Flatten'))
my_model.add(Dense(1024, activation='relu'))
my_model.add(Dropout(0.2))
my_model.add(Dense(num_classes,activation='softmax'))

my_model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

my_model.summary()
for layer in my_model.layers:
    print(layer.name)
print(my_model.get_config())

my_model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs, validation_data=(x_test,y_test))

my_model.save_weights('my_model_weights.h5')

layer_name = 'Flatten' #获取层的名称
intermediate_layer_model = Model(inputs=my_model.input,outputs=my_model.get_layer(layer_name).output)#创建的新模型
# intermediate_output = intermediate_layer_model.predict(x_test_cnn)

with open('intermediate_layer_model.json','w') as f:
    f.write(intermediate_layer_model.to_json())
intermediate_layer_model.save('cnn_weights.h5')
