from keras.datasets import mnist
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from dnn_locate import loc_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

input_shape, labels = (28, 28, 1), 10

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

## define models
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Add, Multiply
from keras.layers import Activation, MaxPooling2D, Conv2D

initializer = initializers.glorot_uniform(seed=0)

## define the detector before TRELU activation
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
discriminator = Sequential()
discriminator.add(Conv2D(32, (3, 3),
		activation='relu', name='last_conv',
		kernel_initializer=initializer,
		bias_initializer=initializer,
		kernel_regularizer=tf.keras.regularizers.l1(0.001),
		bias_regularizer=tf.keras.regularizers.l1(0.001),
		input_shape=input_shape))
discriminator.add(MaxPooling2D((2, 2)))
discriminator.add(Flatten())
discriminator.add(Dense(100, activation='relu',
	kernel_regularizer=tf.keras.regularizers.l1(0.001),
	bias_regularizer=tf.keras.regularizers.l1(0.001),
	kernel_initializer=initializer))
discriminator.add(Dense(labels, activation='softmax',
	kernel_initializer=initializer,
	kernel_regularizer=tf.keras.regularizers.l1(0.001),
	bias_regularizer=tf.keras.regularizers.l1(0.001),
	bias_initializer=initializer))
discriminator.compile(loss='sparse_categorical_crossentropy', 
						optimizer=Adam(lr=0.001),
						metrics=['accuracy'])

## define framework
tau_range = [5., 10.]
shiing = loc_model(input_shape=input_shape,
				discriminator=discriminator,
				tau_range=tau_range,
				task='classification')

es_detect1 = ReduceLROnPlateau(monitor="loss", factor=0.382, min_lr=.0001,
					verbose=1, patience=5, mode="min")
es_detect2 = EarlyStopping(monitor='loss', mode='min', min_delta=.0001, 
						verbose=1, patience=15, restore_best_weights=True)
es_learn = EarlyStopping(monitor='val_accuracy', mode='max', 
						verbose=1, patience=10, restore_best_weights=True)

print('###'*20)
print('###'*5+' '*6+'Load learner'+' '*5+'###'*5)
print('###'*20)

# learn_tmp = shiing.discriminator.fit(x=X_train, y=y_train, 
#									callbacks=[es_learn], epochs=50, 
#									batch_size=128, validation_split=.2)

# shiing.discriminator.save_weights("./saved_model/model1107.h5")
# shiing.discriminator.load_weights("./saved_model/model1107.h5")
shiing.discriminator.load_weights("../tests/saved_model/model1119.h5")
# shiing.discriminator.load_weights("./saved_model/model1126.h5")

print('###'*20)
print('#'*16+' '*5+'Train detector'+' '*5+'#'*16)
print('###'*20)

## fit detector for a range of tau
fit_params={'callbacks': [es_detect1, es_detect2],
			'epochs': 1000, 'batch_size': 64}

shiing.fit(X_train=X_train, y_train=y_train, detector=detector, 
			optimizer=SGD(lr=1.), fit_params=fit_params)

## Visualize the results 
shiing.R_sqaure_path()
shiing.path_plot()
shiing.DA_plot(X=X_test, y=y_test)
