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
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GaussianNoise

input_shape, labels = (28, 28, 1), 10
sample = GaussianNoise(0.2)

method = 'mask'
## load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.
X_test = X_test / 255.

# if method == 'mask':
# 	X_train = X_train / 255.
# else:
# 	X_train = X_train / 127.5 - 1.

ind_set = np.array([i for i in range(len(y_train)) if y_train[i] in [7, 9]])
ind_set_test = np.array([i for i in range(len(y_test)) if y_test[i] in [7, 9]])

X_train, y_train = X_train[ind_set], y_train[ind_set]
X_test, y_test = X_test[ind_set_test], y_test[ind_set_test]

X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

lam = 0.008
elk = LocalGAN(img_shape=input_shape, labels=labels, lam=lam, method=method)
es_detect = EarlyStopping(monitor='loss', mode='min', min_delta=.0001, verbose=1, patience=3, restore_best_weights=True)
es_learn = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5, restore_best_weights=True)

print('###'*20)
print('###'*5+' '*6+'Train for learner'+' '*5+'###'*5)
print('###'*20)

# learn_tmp = elk.discriminator.fit(x=X_train, y=y_train, callbacks=[es_learn], epochs=50, batch_size=128, validation_split=.2)
# elk.discriminator.save_weights("model_01.h5")
# elk.discriminator.load_weights("model_test.h5")
# elk.discriminator.load_weights("model_01.h5")
elk.discriminator.load_weights("model_noise.h5")
train_loss_base, train_acc_base = elk.discriminator.evaluate(X_train, y_train)
test_loss_base, test_acc_base = elk.discriminator.evaluate(X_test, y_test)

print('Baseline: train_loss: %.3f; test_loss: %.3f' %(train_loss_base, test_loss_base))
print('Baseline: train_acc: %.3f; test_acc: %.3f' %(train_acc_base, test_acc_base))


print('###'*20)
print('#'*16+' '*5+'Train for detector'+' '*5+'#'*16)
print('###'*20)

detect_tmp = elk.combined.fit(x=X_train, y=y_train, callbacks=[es_detect], epochs=300, batch_size=128, validation_split=.2)

## show the results
X_train_noise = elk.detector.predict(X_train)
X_test_noise = elk.detector.predict(X_test)

train_loss, train_acc = elk.discriminator.evaluate(X_train_noise, y_train)
test_loss, test_acc = elk.discriminator.evaluate(X_test_noise, y_test)
print('lam: %.5f; train_loss: %.3f; test_loss: %.3f' %(lam, train_loss, test_loss))
print('lam: %.5f; train_acc: %.3f; test_acc: %.3f' %(lam, train_acc, test_acc))


if method == 'mask':
	## show the plot for image 9
	fig = plt.figure(figsize = (10,10))
	ax1 = fig.add_subplot(1,3,1)
	ax1.imshow(X_test[0,:,:,0])

	ax3 = fig.add_subplot(1,3,3)
	ax3.imshow(np.nan_to_num( (X_test_noise[0,:,:,0] - X_test[0,:,:,0]) / X_test[0,:,:,0]))

	ax2 = fig.add_subplot(1,3,2)
	ax2.imshow(X_test_noise[0,:,:,0])

	## show the plot for image 7
	fig = plt.figure(figsize = (10,10))
	ax1 = fig.add_subplot(1,3,1)
	ax1.imshow(X_test[1,:,:,0])

	ax3 = fig.add_subplot(1,3,3)
	ax3.imshow(np.nan_to_num( (X_test_noise[1,:,:,0] - X_test[1,:,:,0]) / X_test[1,:,:,0]))

	ax2 = fig.add_subplot(1,3,2)
	ax2.imshow(X_test_noise[1,:,:,0])

	## show the plot for image 7
	fig = plt.figure(figsize = (10,10))
	ax1 = fig.add_subplot(1,3,1)
	ax1.imshow(X_test[3,:,:,0])

	ax3 = fig.add_subplot(1,3,3)
	ax3.imshow(np.nan_to_num( (X_test_noise[3,:,:,0] - X_test[3,:,:,0]) / X_test[3,:,:,0]))

	ax2 = fig.add_subplot(1,3,2)
	ax2.imshow(X_test_noise[3,:,:,0])

	plt.show()


	## print total sum
	## compute the total mount of the proporion\n",
	print(np.sum(np.nan_to_num( (X_test[0,:,:,0] - X_test_noise[0,:,:,0]) / X_test[0,:,:,0])))
else:
	## show the plot for image 9
	fig = plt.figure(figsize = (10,10))
	ax1 = fig.add_subplot(1,3,1)
	ax1.imshow(X_test[0,:,:,0])

	ax3 = fig.add_subplot(1,3,3)
	ax3.imshow((X_test_noise[0,:,:,0] - X_test[0,:,:,0]))

	ax2 = fig.add_subplot(1,3,2)
	ax2.imshow(X_test_noise[0,:,:,0])

	## show the plot for image 7
	fig = plt.figure(figsize = (10,10))
	ax1 = fig.add_subplot(1,3,1)
	ax1.imshow(X_test[1,:,:,0])

	ax3 = fig.add_subplot(1,3,3)
	ax3.imshow( X_test_noise[1,:,:,0] - X_test[1,:,:,0])

	ax2 = fig.add_subplot(1,3,2)
	ax2.imshow(X_test_noise[1,:,:,0])

	## show the plot for image 7
	fig = plt.figure(figsize = (10,10))
	ax1 = fig.add_subplot(1,3,1)
	ax1.imshow(X_test[3,:,:,0])

	ax3 = fig.add_subplot(1,3,3)
	ax3.imshow(X_test_noise[3,:,:,0] - X_test[3,:,:,0])

	ax2 = fig.add_subplot(1,3,2)
	ax2.imshow(X_test_noise[3,:,:,0])

	plt.show()


	## print total sum
	## compute the total mount of the proporion\n",
	print(np.sum(np.abs(X_test[0,:,:,0] - X_test_noise[0,:,:,0])))