"""
Discriminative feature localization for deep learning models
"""

# Author: Ben Dai <bdai@umn.edu>

import tensorflow as tf
from keras.layers import Input, Dense, Activation, Reshape, Flatten, Add, Multiply
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import sys
import numpy as np

def TRelu(x, max_value=1.):
	return backend.relu(x, max_value=max_value)

def neg_sparse_categorical_crossentropy(y_true, y_pred):
	"""
	negative sparse categorical crossentropy for classification.
	"""
	return -tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

def neg_MSE(y_true, y_pred):
	"""
	negative mean square error for regression.
	"""
	return -tf.keras.losses.MSE(y_true, y_pred)

class Dnn_Locate():
	"""
	class for discriminative feature detection for deep learning models.
	
	Parameters
	----------

	input_shape : tuple-like (shape of the feature/image)
	 For example, in MNIST dataset ``input_shape = (28, 28, 1)``.

	task: {'classification', 'regression'}, default = 'classification'

	discriminator: {keras-defined neural network}
	 A pretrained neural network needs to be explained.

	detector: {keras-defined neural network}
	 A neural network for detector.
	
	R_square: list-like
	 Record for R_sqaure values based on a dataset.

	"""
	def __init__(self, input_shape, discriminator, tau_range, task='classification'):
		# self.labels = labels
		self.input_shape = input_shape
		self.task = task
		self.tau_range = tau_range
		self.discriminator = discriminator
		self.detector = None
		self.combined = None

		# labels: integer, (number of labels for classification, dimension of outcome for regression)
		#  For example, in MNIST dataset ``labels = 10``.

	def build_detector(self, detector, tau=10., max_value=1.):
		"""
		Building a detector for the proposed framework

		Parameters
		----------
		
		detector: {keras-defined neural network}
	 	 A neural network for detector.

	 	tau: {float}
	 	 magnitude of the dector

		"""
		X = Input(shape=self.input_shape)
		noise = detector(X)
		noise = tau*noise
		# noise = Activation(TRelu)(noise)
		noise = backend.relu(noise, max_value=max_value)
		X_mask = Multiply()([noise, X])
		X_noise = Add()([X, -X_mask])
		self.detector = Model(X, X_noise)

	def build_combined(self, optimizer=SGD(lr=.0005)):
		"""
		Building a detector and a combined model for the proposed framework

		Parameters
		----------
		
		detector: {keras-defined neural network}
	 	 A neural network for detector.

		optimizer: {keras-defined optimizer: ``tensorflow.keras.optimizers``}, default = 'SGD(lr=.0005)'
	 	 A optimizer used to train the detector.
	 	 
		"""

		# The decetor generate X_noise based on input imgs
		X = Input(shape=self.input_shape)
		X_noise = self.detector(X)

		# For the combined model we will only train the detector
		self.discriminator.trainable = False

		# The discriminator takes noised images as input and determines probs
		prob = self.discriminator(X_noise)

		# The combined model  (stacked detector and discriminator)
		# Trains the detector to attack the discriminator
		self.combined = Model(X, prob)

		if self.task == 'classification':
			self.combined.compile(loss=neg_sparse_categorical_crossentropy, 
								optimizer=optimizer,
								metrics=['accuracy'])
		
		elif self.task == 'regression':
			self.combined.compile(loss=neg_MSE, optimizer=optimizer)
		else:
			print('the formulation only work for regression and classification.')


	def fit(self, X, y, detector, fit_params, optimizer=SGD(lr=.0005)):
		"""
		Fitting the detector based on a given dataset.

		Parameters
		----------

		X : {array-like} of shape (n_samples, dim_features)
	 	 Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
			 If X is vectorized feature, ``shape`` should be ``(#Samples, dim of feaures)``
			 If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``, that is, **X must channel_last image data**.	

		y : {array-like} of shape (n_samples,)
		 Output vector/matrix relative to X.

		detector : {keras-defined neural network}
	 	 A neural network for detector.
	
		fit_params: {dict of fitting parameters}
	 		See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``, ``validation_split``, ``validation_data``, and so on.
	
		"""
		for tau_tmp in self.tau_range:
			self.build_detector(detector=detector, tau=tau)
			self.build_combined(optimizer=optimizer)
			self.combined.fit(X, y, **fit_params)

	def diff_X(self, X):
		"""
		Return the attacking features generated by the detector.

		Parameters
		----------

		X : {array-like} of shape (n_samples, dim_features)
	 		Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
			 If X is vectorized feature, ``shape`` should be ``(#Samples, dim of feaures)``
			 If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``, that is, **X must channel_last image data**.	
		
		Return
		------
		X_diff : {array-like} of shape (n_samples, dim_features)
			Attacking features generated by the detector
		"""

		X_noise = self.detector.predict(X)
		X_diff = (X_noise - X) / (X + 1e-8)
		return X_diff


	def R_sqaure(self, X, y):
		"""
		Report R_sqaure for the fitted detector based on a given dataset
		
		Parameters
		----------

		X : {array-like} of shape (n_samples, dim_features)
	 		Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
			 If X is vectorized feature, ``shape`` should be ``(#Samples, dim of feaures)``
			 If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``.	

		y : {array-like} of shape (n_samples,)
			 Output vector/matrix relative to X.

		Return
		------
		
		R_sqaure: array of float [0, 1]
			The R_sqaure for fitted detector based on a given dataset.
		"""
		loss_base, acc_base = self.discriminator.evaluate(X, y)
		X_noise = self.detector.predict(X)
		loss_noise, acc_noise = self.discriminator.evaluate(X_noise, y)
		R_square = 1. - loss_base / loss_noise

		return R_square

