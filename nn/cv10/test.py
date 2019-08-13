from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
from keras.models import Model

img_rows, img_cols = 32, 32
channels = 3
num_classes = 10
batch_size = 16
epochs = 10

initial_model = ResNet50(include_top=False,input_shape=(img_rows, img_cols, channels))
x = Flatten()(initial_model.output)
x = Dense(num_classes, activation='softmax')(x)
model = Model(initial_model.input, x)
print(model.summary())
