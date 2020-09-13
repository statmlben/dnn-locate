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
from keras import backend as K

from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import max_norm


class TabNoise(Layer):

	def __init__(self, output_dim, regularizer=tf.keras.regularizers.L1(1.), **kwargs):
	   self.output_dim = output_dim
	   super(TabNoise, self).__init__(**kwargs)
	   # self.regularizer = regularizers.get(regularizer)
	   self.regularizer = regularizer
	   
	def build(self, input_shape):
	   self.D_ = self.add_weight(name='delta', 
	   								shape=(self.output_dim,),
	   								regularizer=self.regularizer,
	   								# constraint=max_norm(max_value=2, axis=0),
	   								initializer=initializers.RandomNormal(seed=0), 
	   								trainable=True)
	   super(TabNoise, self).build(input_shape)

	def call(self, X):
	   # return K.sum([X, -self.D_])
	   return X - K.tanh(self.D_)

	def compute_output_shape(self):
	   return self.output_dim


def build_detector(input_dim, regularizer, type_='mask'):
	model = Sequential()
	model.add(tf.keras.Input(shape=input_dim))
	model.add(TabNoise(input_dim, regularizer=regularizer))
	# if type_ == 'mask':
	# 	model.add(Activation(activations.tanh))
	# else:
	# 	model.add(Activation(activations.sigmoid))

	return model

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