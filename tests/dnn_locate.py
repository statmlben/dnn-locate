from __future__ import print_function, division
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add, Multiply, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, ReLU
from tensorflow.keras.models import Sequential, Model
from keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from keras import initializers

import matplotlib.pyplot as plt
import sys
import numpy as np

lr, inv_temp = 0.0005, 1.

def neg_sparse_categorical_crossentropy(y_true, y_pred):
	return -tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

def neg_MSE(y_true, y_pred):
	return -tf.keras.losses.MSE(y_true, y_pred)

def entropy_loss(y_true, y_pred):
	return tf.keras.losses.categorical_crossentropy(y_pred, y_pred)

class entropy_reg(regularizers.Regularizer):

	def __init__(self, strength):
		self.strength = strength

	def __call__(self, x):
		prob = tf.keras.backend.softmax(tf.keras.backend.abs(x))
		log_prob = tf.keras.backend.log(prob+1e-5)
		# prob = x
		# log_prob = tf.keras.backend.log(prob+1e-5)
		return - self.strength * tf.keras.backend.sum(prob * log_prob)
		# return self.strength*tf.keras.losses.categorical_crossentropy(prob, prob)

class LocalGAN():
	def __init__(self, input_shape, labels, discriminator, detector, optimizer=SGD(lr=.0005), task='classification'):
		self.labels = labels
		self.input_shape = input_shape
		self.optimizer = optimizer
		self.task = task
		self.discriminator = discriminator
		self.detector = detector
		self.R_square_test = []
		self.R_square_train = []

		# The decetor generate noise based on input imgs
		X = Input(shape=self.input_shape)
		X_noise = self.detector(X)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The discriminator takes noised images as input and determines probs
		prob = self.discriminator(X_noise)

		# The combined model  (stacked detector and discriminator)
		# Trains the detector to destroy the discriminator
		self.combined = Model(X, prob)
		if self.task == 'classification':
			self.combined.compile(loss=neg_sparse_categorical_crossentropy, 
				optimizer=optimizer,
				metrics=['accuracy'])
		elif self.task == 'regression':
			self.combined.compile(loss=neg_MSE, 
				optimizer=optimizer)
		else:
			print('the formulation only work for regression and classification.')
		# self.combined.compile(loss=entropy_loss, optimizer=Adam(0.0001), metrics=['accuracy'])