import glob
import numpy as np
import pandas as pd
import os
import shutil 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import tensorflow.keras
"""
files = glob.glob('./train/*.jpg')
cat_files = [fn for fn in files if 'cat' in fn] 
dog_files = [fn for fn in files if 'dog' in fn] 
print(len(cat_files))
print(len(dog_files))
cat_train = np.random.choice(cat_files, size=1500, replace=False) 
dog_train = np.random.choice(dog_files, size=1500, replace=False)
cat_files = list(set(cat_files)-set(cat_train))
dog_files = list(set(dog_files)-set(dog_train))
print(len(cat_files))
print(len(dog_files))
cat_val = np.random.choice(cat_files, size=500, replace=False) 
dog_val = np.random.choice(dog_files, size=500, replace=False) 
cat_files = list(set(cat_files)-set(cat_val)) 
dog_files = list(set(dog_files)-set(dog_val))
print(len(cat_files))
print(len(dog_files))
cat_test = np.random.choice(cat_files, size=500, replace=False) 
dog_test = np.random.choice(dog_files, size=500, replace=False)
print("Cat datasets:", cat_train.shape, cat_val.shape, cat_test.shape) 
print("Dog datasets:", dog_train.shape, dog_val.shape, dog_test.shape)"""

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras

restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(244,300,3))
model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=(244,300,3)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()