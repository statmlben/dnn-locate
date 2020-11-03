from keras.datasets import mnist
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from models.cnn_models import build_detector, build_discriminator
from models.cnn_models_v2 import build_detector, build_discriminator, build_discriminator_gap
from tensorflow.keras import models
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
import numpy as np
from dnn_locate import LocalGAN
from keras.preprocessing.image import ImageDataGenerator
from EDA import show_samples, R_sqaure_path
from keras import backend as K
import tensorflow as tf
import cv2

input_shape, labels = (28, 28, 1), 10

method = 'mask'
## load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.
X_test = X_test / 255.

ind_set = np.array([i for i in range(len(y_train)) if y_train[i] in [7, 9]])
ind_set_test = np.array([i for i in range(len(y_test)) if y_test[i] in [7, 9]])

X_train, y_train = X_train[ind_set], y_train[ind_set]
X_test, y_test = X_test[ind_set_test], y_test[ind_set_test]

X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

demo_ind = np.array([np.where(y_test==7)[0][2], np.where(y_test==9)[0][2]])
discriminator = build_discriminator(img_shape=input_shape, labels=labels)
discriminator.compile(loss='sparse_categorical_crossentropy', 
						optimizer=Adam(lr=0.001),
						metrics=['accuracy'])
es_learn = EarlyStopping(monitor='val_accuracy', mode='max', 
						verbose=1, patience=10, restore_best_weights=True)
learn_tmp = discriminator.fit(x=X_train, y=y_train, callbacks=[es_learn], epochs=50, batch_size=128, validation_split=.2)

# We want to get the final Convolutional Layer
last_conv_layer = discriminator.get_layer('last_conv')
heatmap_model = models.Model([discriminator.inputs], [last_conv_layer.output, discriminator.output])

th_lst, X_test_R, X_test_noise_R = [], [], []
for th_tmp in [1e-3, .2, .3, .4, .5, .6, .7, .8, .9]:
	th_lst.append(th_tmp)
	X_test_R_tmp, X_test_noise_R_tmp = [], []
	for X_test_tmp in X_test[demo_ind]:
		with tf.GradientTape() as gtape:
			conv_output, predictions = heatmap_model(X_test_tmp.reshape(1,28,28,1))
			output_layer = predictions[:, np.argmax(predictions[0])]
			grads = gtape.gradient(output_layer, conv_output)
			pooled_grads = K.mean(grads, axis=(0, 1, 2))

		heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
		heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
		heatmap_reshape = cv2.resize(heatmap[0], (input_shape[:2]))
		X_noise_tmp = X_test_tmp - heatmap_reshape.reshape(28, 28, 1)
		X_test_R_tmp.append(X_test_tmp)
		X_test_noise_R_tmp.append(X_noise_tmp)
	X_test_R.append(X_test_R_tmp)
	X_test_noise_R.append(X_test_noise_R_tmp)

th_lst, X_test_R, X_test_noise_R = np.array(th_lst), np.array(X_test_R), np.array(X_test_noise_R)
show_samples(th_lst, X_test_R, X_test_noise_R, threshold=th_lst, method='noise')

# We load the original image

# with tf.GradientTape() as gtape:
# 	conv_output, predictions = heatmap_model(img_tensor)
# 	loss = predictions[:, np.argmax(predictions[0])]
# 	grads = gtape.gradient(loss, conv_output)
# 	pooled_grads = K.mean(grads, axis=(0, 1, 2))