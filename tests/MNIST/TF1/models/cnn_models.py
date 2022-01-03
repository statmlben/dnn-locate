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

def build_detector(img_shape, lam, type_='mask'):
	# model = Sequential()
	# model.add(Conv2D(32, (2,2), 
	# 			padding="same",
	# 			input_shape=self.img_shape,
	# 			kernel_initializer=initializers.glorot_uniform(seed=0), 
	# 			bias_initializer=initializers.glorot_uniform(seed=0)))
	# model.add(Flatten())
	# model.add(Dense(256, 
	# 		activation ='relu',
	# 		kernel_initializer=initializers.glorot_uniform(seed=0), 
	# 		bias_initializer=initializers.glorot_uniform(seed=0)))
	# model.add(Dense(np.prod(self.img_shape), 
	# 		activation ='relu', 
	# 		kernel_initializer=initializers.glorot_uniform(seed=0),
	# 		bias_initializer=initializers.glorot_uniform(seed=0)))
	# model.add(Reshape(self.img_shape))
	# model.add(BatchNormalization(momentum=0.8))
	# model.add(Conv2D(1, kernel_size=(2,2), padding="same", 
	# 		activation ='sigmoid',
	# 		activity_regularizer=tf.keras.regularizers.L1(self.lam),
	# 		kernel_initializer=initializers.glorot_uniform(seed=0),
	# 		bias_initializer=initializers.glorot_uniform(seed=0)))

	model = Sequential()
	model.add(Conv2D(16, (2,2),
		padding="same",
		input_shape=img_shape,
		kernel_initializer=initializers.glorot_uniform(seed=0), 
		bias_initializer=initializers.glorot_uniform(seed=0)))
	model.add(Flatten())
	model.add(Dense(16, activation='relu', 
		kernel_initializer=initializers.glorot_uniform(seed=0), 
		bias_initializer=initializers.glorot_uniform(seed=0)))
	# model.add(BatchNormalization(momentum=0.8))
	# model.add(Dense(32, activation='relu', 
	# 	kernel_initializer=initializers.glorot_uniform(seed=0), 
	# 	bias_initializer=initializers.glorot_uniform(seed=0)))
	# model.add(BatchNormalization(momentum=0.8))
	model.add(Dense(16, activation='relu',
		kernel_initializer=initializers.glorot_uniform(seed=0), 
		bias_initializer=initializers.glorot_uniform(seed=0)))
	# model.add(BatchNormalization(momentum=0.8))
	# https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8
	if type_ == 'mask':
		model.add(Dense(np.prod(img_shape), 
			# activation = tf.keras.activations.relu(max_value=1),
			activation ='sigmoid',
			# activation ='softmax',
			activity_regularizer=tf.keras.regularizers.L1(lam),
			# activity_regularizer=entropy_reg(self.lam),
			kernel_initializer=initializers.glorot_uniform(seed=0),
			bias_initializer=initializers.glorot_uniform(seed=0)))
		# model.add(ReLU(max_value=1))
	else:
		model.add(Dense(np.prod(img_shape), 
				activation ='tanh',
				kernel_initializer=initializers.glorot_uniform(seed=0),
				activity_regularizer=tf.keras.regularizers.L1(lam),
				# activity_regularizer = entropy_reg(self.lam),
				bias_initializer=initializers.glorot_uniform(seed=0)))

	# model.add(ReLU(threshold=.1))
	model.add(Reshape(img_shape))

	img = Input(shape=img_shape)
	noise = model(img)
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
			activation='relu', 
			kernel_initializer=initializers.glorot_uniform(seed=0),
			bias_initializer=initializers.glorot_uniform(seed=0),
			kernel_regularizer=tf.keras.regularizers.L1(0.001),
			bias_regularizer=tf.keras.regularizers.L1(0.001),
			input_shape=img_shape))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu',
		kernel_regularizer=tf.keras.regularizers.L1(0.001),
		bias_regularizer=tf.keras.regularizers.L1(0.001),
		kernel_initializer=initializers.glorot_uniform(seed=0)))
	model.add(Dense(labels, activation='softmax', 
		kernel_initializer=initializers.glorot_uniform(seed=0),
		kernel_regularizer=tf.keras.regularizers.L1(0.001),
		bias_regularizer=tf.keras.regularizers.L1(0.001),
		bias_initializer=initializers.glorot_uniform(seed=0)))
	# model.summary()

	img = Input(shape=img_shape)
	prob = model(img)
	return Model(img, prob)