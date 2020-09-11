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

import matplotlib.pyplot as plt
import sys
import numpy as np

from keras_gradient_noise import add_gradient_noise
from SGLD import SGLD

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
	def __init__(self, input_shape, labels, discriminator, detector, optimizer=SGD(lr=0.0005), task='classification'):
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


	# def train(self, epochs, batch_size=128, sample_interval=50):

	# 	# Load the dataset
	# 	(X_train, y_train), (_, _) = mnist.load_data()

	# 	# Rescale -1 to 1
	# 	X_train = X_train / 255.0
	# 	X_train = np.expand_dims(X_train, axis=3)

	# 	for epoch in range(epochs):

	# 		# ---------------------
	# 		#  Train Discriminator
	# 		# ---------------------

	# 		# Select a random batch of images
	# 		idx = np.random.randint(0, X_train.shape[0], batch_size)
	# 		imgs = X_train[idx]

	# 		# noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

	# 		# Generate a batch of new images
	# 		if epoch == 0:
	# 			imgs_noise = imgs
	# 		else:
	# 			imgs_noise = self.detector.predict(imgs)

	# 		# Train the discriminator
	# 		d_loss_real = self.discriminator.train_on_batch(imgs_noise, y_train[idx])
	# 		# d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
	# 		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

	# 		# ---------------------
	# 		#  Train Detector
	# 		# ---------------------

	# 		noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

	# 		# Train the generator (to have the discriminator label samples as valid)
	# 		g_loss = self.combined.train_on_batch(noise, valid)

	# 		# Plot the progress
	# 		print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

	# 		# If at save interval => save generated image samples
	# 		if epoch % sample_interval == 0:
	# 			self.sample_images(epoch)

	# def sample_images(self, epoch):
	# 	r, c = 5, 5
	# 	noise = np.random.normal(0, 1, (r * c, self.latent_dim))
	# 	gen_imgs = self.generator.predict(noise)

	# 	# Rescale images 0 - 1
	# 	gen_imgs = 0.5 * gen_imgs + 0.5

	# 	fig, axs = plt.subplots(r, c)
	# 	cnt = 0
	# 	for i in range(r):
	# 		for j in range(c):
	# 			axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
	# 			axs[i,j].axis('off')
	# 			cnt += 1
	# 	fig.savefig("images/%d.png" % epoch)
	# 	plt.close()