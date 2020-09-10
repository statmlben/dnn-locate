from __future__ import print_function, division
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add, Multiply, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from keras import initializers
import numpy as np

def build_detector(input_dim, lam, type_='mask'):
	model = Sequential()
	if type_ == 'mask':
		model.add(Dense(input_dim, input_dim=1,
			activation ='sigmoid',
			activity_regularizer=tf.keras.regularizers.L1(lam),
			kernel_initializer=initializers.glorot_uniform(seed=0),
			bias_initializer=initializers.glorot_uniform(seed=0)))
	else:
		model.add(Dense(input_dim, input_dim=1,
			activation ='tanh',
			activity_regularizer=tf.keras.regularizers.L1(lam),
			kernel_initializer=initializers.glorot_uniform(seed=0),
			bias_initializer=initializers.glorot_uniform(seed=0)))

	X = Input(shape=1)
	noise = model(X)
	# mask img
	if type_ == 'mask':
		mask_X = Multiply()([noise, X])
		X_noise = Add()([X, -mask_X])
		# img_noise = mask_img
	else:
		X_noise = Add()([X, noise])
	# img = model(noise)
	# model.summary()

	return Model(X, X_noise)

def build_discriminator(input_dim, lam, labels=1):

	model = Sequential()
	model.add(Dense(labels, input_dim=input_dim,
		kernel_regularizer=tf.keras.regularizers.L1(lam),
		bias_regularizer=tf.keras.regularizers.L1(lam),
		kernel_initializer=initializers.glorot_uniform(seed=0)))
	# model.summary()

	X = Input(shape=input_dim)
	prob = model(X)
	return Model(X, prob)