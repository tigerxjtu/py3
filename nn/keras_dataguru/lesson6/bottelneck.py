from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam

InceptionV3_model = InceptionV3(weights='imagenet',include_top=False,input_shape=(150,150,3))

datagen = ImageDataGenerator(
        rotation_range = 40,      # 随机旋转角度
        width_shift_range = 0.2,  # 随机水平平移
        height_shift_range = 0.2, # 随机竖直平移
        rescale = 1./255,         # 数值归一化
        shear_range = 0.2,        # 随机裁剪
        zoom_range  =0.2,         # 随机放大
        horizontal_flip = True,   # 水平翻转
        fill_mode='nearest')      # 填充方式

batch_size = 32
train_steps = int((2000 +  batch_size - 1)/batch_size)*10
test_steps = int((1000 +  batch_size - 1)/batch_size)*10

generator = datagen.flow_from_directory(
        r'dataset/train',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,  # 不生成标签
        shuffle=False)    # 不随机打乱

# 得到训练集数据
bottleneck_features_train = InceptionV3_model.predict_generator(generator, train_steps)
print(bottleneck_features_train.shape)

# 保存训练集bottleneck结果
np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

generator = datagen.flow_from_directory(
       r'dataset/test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None, # 不生成标签
        shuffle=False)  # 不随机打乱
# 得到预测集数据
bottleneck_features_test = InceptionV3_model.predict_generator(generator, test_steps)
print(bottleneck_features_test.shape)
# 保存测试集bottleneck结果
np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test)

train_data = np.load(open('bottleneck_features_train.npy','rb'))
# the features were saved in order, so recreating the labels is easy
labels = np.array([0] * 1000 + [1] * 1000)
train_labels = np.array([])
for _ in range(10):
    train_labels=np.concatenate((train_labels,labels))

test_data = np.load(open('bottleneck_features_test.npy','rb'))
labels = np.array([0] * 500 + [1] * 500)
test_labels = np.array([])
for _ in range(10):
    test_labels=np.concatenate((test_labels,labels))

train_labels = np_utils.to_categorical(train_labels,num_classes=2)
test_labels = np_utils.to_categorical(test_labels,num_classes=2)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# 定义优化器
adam = Adam(lr=1e-4)

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=20, batch_size=batch_size,
          validation_data=(train_data, train_labels))

model.save_weights('bottleneck_fc_model.h5')