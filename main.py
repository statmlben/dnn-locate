from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import sys
import numpy as np
from dnn_locate import LocalGAN

input_shape, labels = (28, 28, 1), 10

method = 'mask'
## load data
(X_train, y_train), (_, _) = mnist.load_data()

if method == 'mask':
	X_train = X_train / 255.
else:
	X_train = X_train / 127.5 - 1.

ind_set = np.array([i for i in range(len(y_train)) if y_train[i] in [7, 9]])
X_train, y_train = X_train[ind_set], y_train[ind_set]
X_train = np.expand_dims(X_train, axis=3)

elk = LocalGAN(img_shape=input_shape, labels=labels, lam=.00, method=method)
epochs = 50
es_detect = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
es_learn = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5, restore_best_weights=True)

learner_weights = elk.discriminator.get_weights()

for epoch in range(epochs):
	if epoch > 0:
		print('###'*20)
		print('#'*16+' '*5+'Train for detector'+' '*5+'#'*16)
		print('###'*20)

		detect_tmp = elk.combined.fit(x=X_train, y=y_train, callbacks=[es_detect], epochs=5, batch_size=128, validation_split=.2)
		X_noise = elk.detector.predict(X_train)
		# print('sum of abs of noise: %.3f' %np.sum(X_noise[:,:,:,0]))
	else:
		X_noise = X_train
	# ind_tmp = np.random.randint(100)
	# plt.imshow(X_noise[ind_tmp,:,:,0])
	# plt.show()

	print('###'*20)
	print('###'*5+' '*5+'Train for learner'+' '*5+'###'*5)
	print('###'*20)
	
	elk.discriminator.set_weights(learner_weights)
	learn_tmp = elk.discriminator.fit(x=X_noise, y=y_train, callbacks=[es_learn], epochs=5, batch_size=128, validation_split=.2)
	# score, acc = elk.discriminator.evaluate(X_train, y_train)
	# print('performance for discriminator: score: %.3f; acc: %.3f' %(score, acc))

	# score, acc = elk.combined.evaluate(X_train, y_train)
	# print('performance for combined: score: %.3f; acc: %.3f' %(score, acc))
	# diff = abs(learn_tmp.history['accuracy'][0] - detect_tmp.history['accuracy'][0]) / abs(learn_tmp.history['accuracy'][0] + 1e-3)
	# print('epoch: %d; diff: %.3f' %(epoch, diff))

	if epoch > 0:
		if (learn_tmp.history['accuracy'][-1] < .5) or (detect_tmp.history['accuracy'][-1] > .9):
			break;
# batch_size, epochs = 128, 100
# for epoch in range(epochs):

# 	# ---------------------
# 	#  Train Discriminator
# 	# ---------------------

# 	# Select a random batch of images
# 	idx = np.random.randint(0, X_train.shape[0], batch_size)
# 	imgs = X_train[idx]

# 	# Generate a batch of new images
# 	if epoch == 0:
# 		imgs_noise = imgs
# 	else:
# 		imgs_noise = elk.detector.predict(imgs)

# 	# Train the discriminator
# 	d_loss = elk.discriminator.train_on_batch(imgs_noise, y_train[idx])
# 	# elk.discriminator.evaluate(imgs_noise, y_train[idx])
# 	# ---------------------
# 	#  Train Detector
# 	# ---------------------

# 	# Train the generator (to have the discriminator label samples as valid)
# 	h_loss = elk.combined.train_on_batch(imgs, y_train[idx])
# 	# elk.combined.evaluate(imgs, y_train[idx])

# 	# Plot the progress
# 	print ("%d [D loss: %f, acc.: %.2f%%] [h loss: %f, acc.: %.2f%%]" 
# 			% (epoch, d_loss[0], 100*d_loss[1], h_loss[0], 100*h_loss[1]))

# 	# If at save interval => save generated image samples
# 	# if epoch % sample_interval == 0:
# 	# 	self.sample_images(epoch)

# 	## check the discriminator in combined: should be equal
# 	# X_noise = elk.detector.predict(X_train[idx])
# 	# elk.discriminator.evaluate(X_noise, y_train[idx])
# 	# elk.combined.evaluate(X_train[idx], y_train[idx])


## plot the figure.
X_tmp = X_train[:100].copy()
X_noise = elk.detector.predict(X_tmp)
noise = X_noise - X_tmp
for j in range(5):
	ind_tmp = np.random.randint(100)
	plt.imshow(X_tmp[ind_tmp,:,:,0])
	plt.show()
	plt.imshow(noise[ind_tmp,:,:,0])
	plt.show()
	plt.imshow(X_noise[ind_tmp,:,:,0])
	plt.show()




