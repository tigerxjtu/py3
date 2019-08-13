import keras
from tensorflow.examples.tutorials.mnist import input_data
import os
from keras.callbacks import TensorBoard

os.environ['KMP_DUPLICATE_LIB_OK']='True'


batch_size = 500
epoch_num = 20

# preparation_data
mnist = input_data.read_data_sets('../cv6/MNIST_data/',one_hot=True)
train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels


my_model = keras.models.Sequential()

#input layer
my_model.add(keras.layers.Dense(512,input_dim=784))
my_model.add(keras.layers.Activation('relu'))
my_model.add(keras.layers.Dropout(0.2))

#hidden layer1
my_model.add(keras.layers.Dense(512))
my_model.add(keras.layers.Activation('relu'))
my_model.add(keras.layers.Dropout(0.2))

#hidden layer2
my_model.add(keras.layers.Dense(512))
my_model.add(keras.layers.Activation('relu'))
my_model.add(keras.layers.Dropout(0.2))

#hidden layer2
my_model.add(keras.layers.Dense(512))
my_model.add(keras.layers.Activation('relu'))
my_model.add(keras.layers.Dropout(0.2))

#hidden layer2
my_model.add(keras.layers.Dense(512))
my_model.add(keras.layers.Activation('relu'))
my_model.add(keras.layers.Dropout(0.2))


#output layer
my_model.add(keras.layers.Dense(10))
my_model.add(keras.layers.Activation('softmax'))

#optimizer

my_model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

#feed data
my_model.fit(x = train_x, y =  train_y, batch_size=batch_size,epochs=epoch_num, validation_data=(test_x,test_y),
             callbacks=[TensorBoard(log_dir='log/mnist1')])
