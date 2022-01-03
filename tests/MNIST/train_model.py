from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from models.cnn_models import build_detector, build_discriminator
from models.cnn_models_v2 import build_detector, build_discriminator
import json
from dnn_locate import LocalGAN
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from EDA import show_samples, R_sqaure_path, show_diff_samples
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
np.random.seed(3)
tf.random.set_seed(3)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255., x_test / 255.

input_shape, labels = (32, 32, 3), 10

## fit the VGG16
from vgg16 import VGG16Model
discriminator = VGG16Model()
optimizer = tf.keras.optimizers.SGD(learning_rate=.1,
                                    decay=1e-6, momentum=.9, nesterov=True)

discriminator.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, 
                        metrics=['accuracy'])

es_detect1 = ReduceLROnPlateau(monitor="val_loss", factor=0.382, min_lr=1e-5,
                    verbose=1, patience=5, mode="min")

es_detect2 = EarlyStopping(monitor='val_loss', mode='min', min_delta=.0001, 
                        verbose=1, patience=15, restore_best_weights=True)

discriminator.fit(x_train, y_train, batch_size=128, epochs=250, validation_split=.2, 
                    callbacks=[es_detect1, es_detect2])

discriminator.save_weights('vgg16.h5')
