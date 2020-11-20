from __future__ import print_function, division
import tensorflow as tf
from keras.datasets import mnist
# from tensorflow.python.keras import backend
from keras import backend
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add, Multiply, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras import initializers
import numpy as np

initializer = initializers.glorot_uniform(seed=0)
# initializer = initializers.Ones()


def TRelu(x):
	return backend.relu(x, max_value=1.)

def build_detector(img_shape, lam, type_='mask'):

	model = Sequential()
	model.add(Conv2D(16, (2,2),
		padding="same",
		input_shape=img_shape,
		kernel_initializer=initializer, 
		bias_initializer=initializer))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', 
		kernel_initializer=initializer, 
		bias_initializer=initializer))
	model.add(Dense(128, activation='relu',
		kernel_initializer=initializer, 
		bias_initializer=initializer))
	if type_ == 'mask':
		model.add(Dense(np.prod(img_shape), 
			# activation ='sigmoid',
			activation ='softmax',
			# activity_regularizer=tf.keras.regularizers.l1(lam),
			# activity_regularizer=entropy_reg(self.lam),
			kernel_initializer=initializer,
			bias_initializer=initializer))
		# model.add(ReLU(max_value=1))
	else:
		model.add(Dense(np.prod(img_shape), 
				activation ='tanh',
				kernel_initializer=initializer,
				# activity_regularizer=tf.keras.regularizers.l1(lam),
				# activity_regularizer = entropy_reg(self.lam),
				bias_initializer=initializer))

	# model.add(ReLU(threshold=.1))
	model.add(Reshape(img_shape))

	img = Input(shape=img_shape)
	noise = model(img)
	noise = lam*noise
	noise = Activation(TRelu)(noise)

	# mask img
	if type_ == 'mask':
		mask_img = Multiply()([noise, img])
		img_noise = Add()([img, -mask_img])
		# img_noise = mask_img
	else:
		img_noise = Add()([img, noise])
	# img = model(noise)
	# model.summary()

	return Model(img, img_noise)

def build_discriminator(img_shape, labels):

	model = Sequential()

	model.add(Conv2D(32, (3, 3),
			activation='relu', name='last_conv',
			kernel_initializer=initializer,
			bias_initializer=initializer,
			kernel_regularizer=tf.keras.regularizers.l1(0.001),
			bias_regularizer=tf.keras.regularizers.l1(0.001),
			input_shape=img_shape))
	
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu',
		kernel_regularizer=tf.keras.regularizers.l1(0.001),
		bias_regularizer=tf.keras.regularizers.l1(0.001),
		kernel_initializer=initializer))
	# model.add(Dense(labels, activation='softmax', name='output_layer',
	# 	kernel_initializer=initializer,
	# 	kernel_regularizer=tf.keras.regularizers.l1(0.001),
	# 	bias_regularizer=tf.keras.regularizers.l1(0.001),
	# 	bias_initializer=initializer))
	model.add(Dense(labels, activation='softmax',
		kernel_initializer=initializer,
		kernel_regularizer=tf.keras.regularizers.l1(0.001),
		bias_regularizer=tf.keras.regularizers.l1(0.001),
		bias_initializer=initializer))
	# model.summary()
	return model


def build_discriminator_gap(img_shape, labels):
	model = Sequential()
	model.add(Conv2D(64, (2, 2),
			activation='relu',
			# padding="same",
			kernel_initializer=initializer,
			bias_initializer=initializer,
			kernel_regularizer=tf.keras.regularizers.l1(0.001),
			bias_regularizer=tf.keras.regularizers.l1(0.001),
			input_shape=img_shape))
	model.add(Conv2D(64, (2, 2),
		name='last_conv',
		activation='relu',
		# padding="same",
		kernel_initializer=initializer,
		bias_initializer=initializer,
		kernel_regularizer=tf.keras.regularizers.l1(0.001),
		bias_regularizer=tf.keras.regularizers.l1(0.001),
		input_shape=img_shape))
	model.add(MaxPooling2D((2, 2)))	
	model.add(GlobalAveragePooling2D())
	# model.add(Flatten())
	# model.add(Dense(100, activation='relu',
	# 	kernel_regularizer=tf.keras.regularizers.l1(0.001),
	# 	bias_regularizer=tf.keras.regularizers.l1(0.001),
	# 	kernel_initializer=initializer))
	model.add(Dense(labels, activation='softmax', name='output_layer',
		kernel_initializer=initializer,
		kernel_regularizer=tf.keras.regularizers.l1(0.001),
		bias_regularizer=tf.keras.regularizers.l1(0.001),
		bias_initializer=initializer))
	# model.summary()
	# img = Input(shape=img_shape)
	# prob = model(img)
	return model