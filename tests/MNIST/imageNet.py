from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from models.cnn_models import build_detector, build_discriminator
from models.cnn_models_v2 import build_detector, build_discriminator
import json
from dnn_locate import loc_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from EDA import show_samples, R_sqaure_path, show_diff_samples
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import initializers
np.random.seed(3)
tf.random.set_seed(3)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255., x_test / 255.

input_shape, labels = (32, 32, 3), 10

from vgg16 import VGG16Model

tau = 10.

## define discriminator and detector_backend
detector_backend = tf.keras.applications.vgg16.VGG16(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=input_shape, pooling=None, classes=1000,
    classifier_activation='relu'
    )

discriminator = VGG16Model()

discriminator.compile(loss='sparse_categorical_crossentropy', 
                        optimizer=Adam(lr=0.001),
                        metrics=['accuracy'])
discriminator.load_weights('./saved_model/vgg16.h5')

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add, Multiply, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D

initializer = initializers.glorot_uniform(seed=0)

## define the backend detector before TRELU activation
detector_backend = Sequential()
detector_backend.add(Conv2D(32, (2,2),
    padding="same",
    input_shape=input_shape,
    kernel_initializer=initializer, 
    bias_initializer=initializer))
detector_backend.add(BatchNormalization())
detector_backend.add(Flatten())
detector_backend.add(Dense(256, activation='relu', 
    kernel_initializer=initializer, 
    bias_initializer=initializer))
detector_backend.add(BatchNormalization())
detector_backend.add(Dense(256, activation='relu',
    kernel_initializer=initializer, 
    bias_initializer=initializer))
detector_backend.add(BatchNormalization())
detector_backend.add(Dense(np.prod(input_shape), 
    activation ='softmax',
    kernel_initializer=initializer,
    bias_initializer=initializer))
detector_backend.add(Reshape(input_shape))


## define framework
es_detect1 = ReduceLROnPlateau(monitor="loss", factor=0.382, min_lr=.0001,
                    verbose=1, patience=5, mode="min")
es_detect2 = EarlyStopping(monitor='loss', mode='min', min_delta=.0001, 
                        verbose=1, patience=15, restore_best_weights=True)

tau_range = [10., 12.]
cue = loc_model(input_shape=input_shape,
                detector_backend=detector_backend,
                discriminator=discriminator,
                tau_range=tau_range,
                task='classification')

fit_params={'callbacks': [es_detect1, es_detect2], 
            'epochs': 1000, 'batch_size': 32}

cue.fit(X_train=x_train, y_train=y_train, demo_ind=range(5),
            optimizer=SGD(lr=1.), fit_params=fit_params)

