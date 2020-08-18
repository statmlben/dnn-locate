from __future__ import print_function, division
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add, Multiply, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, ReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt
import sys
import numpy as np

def neg_sparse_categorical_crossentropy(y_true, y_pred):
	return -tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

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
	def __init__(self, img_shape, labels, lam=0.01, method='mask', optimizer=Adam(0.0005)):
		self.img_rows = img_shape[0]
		self.img_cols = img_shape[1]
		self.channels = img_shape[2]
		self.labels = labels
		self.img_shape = img_shape
		self.lam=lam
		self.optimizer = optimizer
		self.method=method

		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

		# Build the detector
		self.detector = self.build_detector()

		# The decetor generate noise based on input imgs
		img = Input(shape=self.img_shape)
		img_noise = self.detector(img)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The discriminator takes noised images as input and determines probs
		prob = self.discriminator(img_noise)

		# The combined model  (stacked detector and discriminator)
		# Trains the detector to destroy the discriminator
		self.combined = Model(img, prob)
		self.combined.compile(loss=neg_sparse_categorical_crossentropy, optimizer=Adam(0.001), metrics=['accuracy'])
		# self.combined.compile(loss=entropy_loss, optimizer=Adam(0.0001), metrics=['accuracy'])

	def build_detector(self):

		model = Sequential()

		model.add(Flatten(input_shape=self.img_shape))
		model.add(Dense(256, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'))
		# model.add(LeakyReLU(alpha=0.2))
		# model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(256, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'))
		# model.add(LeakyReLU(alpha=0.2))
		# model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(1024, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'))
		# model.add(LeakyReLU(alpha=0.2))
		# model.add(BatchNormalization(momentum=0.8))
		if self.method == 'mask':
			model.add(Dense(np.prod(self.img_shape), 
				# activation = tf.keras.activations.relu(max_value=1),
				activation ='sigmoid',
				activity_regularizer=tf.keras.regularizers.L1(self.lam),
				kernel_initializer='zeros',
				bias_initializer='zeros'))
			# model.add(ReLU(max_value=1))
		else:
			model.add(Dense(np.prod(self.img_shape), activation ='tanh', kernel_initializer='zeros',
				bias_initializer='zeros', activity_regularizer=tf.keras.regularizers.L1(self.lam)))

		model.add(Reshape(self.img_shape))

		# model.summary()

		img = Input(shape=self.img_shape)
		noise = model(img)
		# mask img
		if self.method == 'mask':
			mask_img = Multiply()([noise, img])
			img_noise = Add()([img, -mask_img])
			# img_noise = mask_img
		else:
			img_noise = Add()([img, noise])
		# img = model(noise)

		return Model(img, img_noise)

	def build_discriminator(self):

		model = Sequential()

		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=self.img_shape))
		model.add(MaxPooling2D((2, 2)))
		model.add(Flatten())
		model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(self.labels, activation='softmax'))
		model.summary()

		img = Input(shape=self.img_shape)
		prob = model(img)

		return Model(img, prob)

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


if __name__ == '__main__':
	gan = GAN()
	gan.train(epochs=30000, batch_size=32, sample_interval=200)