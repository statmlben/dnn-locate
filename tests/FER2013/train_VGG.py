import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import confusion_matrix
# from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop,Adam,SGD
from tensorflow_addons.optimizers import AdamW, SGDW
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K
import time
from tensorflow.keras.models import Sequential

# dataset: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

image_array = np.load('../valid_image_array.npy')
label = np.load('../valid_label.npy')
image_array /= 255.
image_array = image_array.astype('float32')

oh_label = to_categorical(label)

datagen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=False,
    featurewise_std_normalization=False,
    brightness_range=(0.8, 1.2),
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2)

datagen.fit(image_array)

from vgg16 import VGG16Model

model_full = VGG16Model()
model_full.compile(optimizer=SGDW(learning_rate=.001, weight_decay=.000, momentum=.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

K.clear_session()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

scheduler = ReduceLROnPlateau(monitor='val_loss', mode='min',
                            factor=0.1, patience=4, verbose=True, min_lr=1e-8)

es = EarlyStopping(monitor='val_loss', mode='min',
                   verbose=1, patience=6,
                   restore_best_weights=True)


model_full.fit(datagen.flow(image_array, oh_label, batch_size=128,
         subset='training'),
         validation_data=datagen.flow(image_array, oh_label,
         batch_size=128, subset='validation'),
         epochs=500,
         callbacks=[scheduler, es])

# model_full.fit(x=image_array, y=oh_label,
#                 batch_size=128, 
#                 epochs=500,
#                 validation_split=.2,
#                 callbacks=[scheduler, es])

model_full.save_weights('./saved_model/vgg16.h5')


model_full.evaluate(image_array, oh_label)