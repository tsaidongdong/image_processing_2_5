from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer,BatchNormalization,Activation,AveragePooling2D,GlobalAveragePooling2D
from keras.models import Sequential
from keras import optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras
import glob
import numpy as np
import pandas as pd
import os
import shutil 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import Model
import tensorflow.keras

# 建立模型
model = Sequential()
# 这里使用卷积神经网络，传入100*100像素的彩色图片，传出时为94*94*32
model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))
# 使用批标准化
model.add(BatchNormalization(axis = 3, name = 'bn0'))
# 激活函数为ReLU（线性整流函数）
model.add(Activation('relu'))
# 对于空间数据的最大池化
model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1, 1), name="conv1"))
model.add(Activation('relu'))
# 对于空间数据的平均池化
model.add(AveragePooling2D((3, 3), name='avg_pool'))

model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.5))
# 编译模型
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(img_width, img_height, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D(name='average_pool')(x)
predictions = Dense(class_num, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-3),
              metrics=['acc'])

