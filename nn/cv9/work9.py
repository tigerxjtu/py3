import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import numpy as np

batch_size = 120
num_classes = 10
epochs = 10
img_shape = (28,28,1)

# input dimensions
img_rows, img_cols = 28,28

# dataset input
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data initialization
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Model (forward)
# Initialize model
alexnet = Sequential()

# Layer 1
alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
      padding='same'))

##################### your codes
alexnet.add(BatchNormalization())
alexnet.add(MaxPooling2D((3, 3), strides=2, padding='valid'))

alexnet.add(Conv2D(128, (5, 5), strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
alexnet.add(BatchNormalization())
alexnet.add(MaxPooling2D((3, 3), strides=2, padding='valid'))

alexnet.add(Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
alexnet.add(Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
alexnet.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
alexnet.add(MaxPooling2D((3, 3), strides=2, padding='valid'))


alexnet.add( Flatten())
alexnet.add(Dense(4096, activation='relu'))
alexnet.add(Dropout(0.5))

alexnet.add(Dense(4096, activation='relu'))
alexnet.add(Dropout(0.5))

# Layer 8
alexnet.add(Dense(num_classes))
alexnet.add(BatchNormalization())
alexnet.add(Activation('softmax'))

# Model (backward)
alexnet.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

# Feed Data
alexnet.fit(x_train, y_train, batch_size= batch_size, epochs= epochs, validation_data=(x_test,y_test))
