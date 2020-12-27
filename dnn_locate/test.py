from keras.datasets import mnist
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from models.cnn_models import build_detector, build_discriminator
# from models.cnn_models_v2 import build_detector, build_discriminator

from dnn_locate import Dnn_Locate
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from EDA import show_samples, R_sqaure_path, show_diff_samples
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

input_shape, labels = (28, 28, 1), 10

method = 'mask'
## load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.
X_test = X_test / 255.

ind_set = np.array([i for i in range(len(y_train)) if y_train[i] in [7, 9]])
ind_set_test = np.array([i for i in range(len(y_test)) if y_test[i] in [7, 9]])

X_train, y_train = X_train[ind_set], y_train[ind_set]
X_test, y_test = X_test[ind_set_test], y_test[ind_set_test]

X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

demo_ind = np.array([np.where(y_test==7)[0][0], np.where(y_test==9)[0][17]])
## define models
tau=10

from keras import initializers
from keras.models import Sequential, Model
from keras.layers import UpSampling2D, Conv2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add, Multiply, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D

initializer = initializers.glorot_uniform(seed=0)

detector = Sequential()
detector.add(Conv2D(32, (2,2),
	padding="same",
	input_shape=input_shape,
	kernel_initializer=initializer, 
	bias_initializer=initializer))
detector.add(Flatten())
detector.add(Dense(128, activation='relu', 
	kernel_initializer=initializer, 
	bias_initializer=initializer))
detector.add(Dense(128, activation='relu',
	kernel_initializer=initializer, 
	bias_initializer=initializer))
detector.add(Dense(np.prod(input_shape), 
	activation ='softmax',
	kernel_initializer=initializer,
	bias_initializer=initializer))
detector.add(Reshape(input_shape))

## define discriminator
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

	model.add(Dense(labels, activation='softmax',
		kernel_initializer=initializer,
		kernel_regularizer=tf.keras.regularizers.l1(0.001),
		bias_regularizer=tf.keras.regularizers.l1(0.001),
		bias_initializer=initializer))
	# model.summary()
	return model

discriminator = build_discriminator(img_shape=input_shape, labels=labels)
discriminator.compile(loss='sparse_categorical_crossentropy', 
						optimizer=Adam(lr=0.001),
						metrics=['accuracy'])

## define framework
shiing = Dnn_Locate(input_shape=input_shape,
				discriminator=discriminator,
				# optimizer=SGD(lr=10./tau),
				task='classification')

shiing.build_detector(detector=detector, tau=tau)
shiing.build_combined(optimizer=SGD(lr=10./tau))

# es_detect1 = ReduceLROnPlateau(monitor="loss", factor=0.382, min_lr=0.0001, 
# 						verbose=1, patience=4, mode="min")
es_detect1 = ReduceLROnPlateau(monitor="loss", factor=0.382, min_lr=.0001,
					verbose=1, patience=5, mode="min")
es_detect2 = EarlyStopping(monitor='loss', mode='min', min_delta=.0001, 
						verbose=1, patience=15, restore_best_weights=True)

es_learn = EarlyStopping(monitor='val_accuracy', mode='max', 
						verbose=1, patience=10, restore_best_weights=True)

print('###'*20)
print('###'*5+' '*6+'Load learner'+' '*5+'###'*5)
print('###'*20)

# learn_tmp = shiing.discriminator.fit(x=X_train, y=y_train, callbacks=[es_learn], epochs=50, batch_size=128, validation_split=.2)
# shiing.discriminator.save_weights("./saved_model/model1107.h5")
# shiing.discriminator.load_weights("./saved_model/model1107.h5")
shiing.discriminator.load_weights("../tests/saved_model/model1119.h5")
# shiing.discriminator.load_weights("./saved_model/model1126.h5")

print('###'*20)
print('#'*16+' '*5+'Train detector'+' '*5+'#'*16)
print('###'*20)

fit_params={'callbacks': [es_detect1, es_detect2], 
			'epochs': 1000, 'batch_size': 64}

shiing.fit(X=X_train, y=y_train, optimizer=SGD(lr=10./tau), fit_params=fit_params)

detect_tmp = shiing.combined.fit(x=X_train, y=y_train, optimizer=SGD(lr=10./tau), 
								callbacks=[es_detect1, es_detect2], 
								epochs=1000, batch_size=128)
# validation_split=.2

## show the results
X_train_noise = shiing.detector.predict(X_train)
X_test_noise = shiing.detector.predict(X_test)

train_loss, train_acc = shiing.discriminator.evaluate(X_train_noise, y_train)
test_loss, test_acc = shiing.discriminator.evaluate(X_test_noise, y_test)
print('tau: %.2f; train_loss: %.3f; test_loss: %.3f' %(tau, train_loss, test_loss))
print('tau: %.2f; train_acc: %.3f; test_acc: %.3f' %(tau, train_acc, test_acc))


R_sqaure_path(lam_range, norm_lst, norm_test_lst,
				R_square_train_lst, R_square_test_lst)

X_test_R, X_test_noise_R, R_square_test_lst = np.array(X_test_R), np.array(X_test_noise_R), np.array(R_square_test_lst)
show_samples(R_square_test_lst, X_test_R, X_test_noise_R)

# X_test_demo, X_test_noise_demo = [], []
# for i in range(5,14):
# 	demo_ind = np.array([np.where(y_test==7)[0][i], np.where(y_test==9)[0][i]])
# 	X_test_demo.append(X_test[demo_ind])
# 	X_test_noise_demo.append(X_test_noise[demo_ind])
# X_test_demo, X_test_noise_demo = np.array(X_test_demo), np.array(X_test_noise_demo)
# show_diff_samples(X_test=X_test_demo, X_test_noise=X_test_noise_demo)
