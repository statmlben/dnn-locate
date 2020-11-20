from keras.datasets import mnist
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from models.cnn_models import build_detector, build_discriminator
from models.cnn_models_v2 import build_detector, build_discriminator

import matplotlib.pyplot as plt
from dnn_locate import LocalGAN
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from EDA import show_samples, R_sqaure_path, show_diff_samples
import tensorflow as tf

np.random.seed(3)
# tf.random.set_seed(3)

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
# demo_ind = np.array([np.where(y_test==7)[0][2], np.where(y_test==9)[0][2]])

discriminator = build_discriminator(img_shape=input_shape, labels=labels)
discriminator.compile(loss='sparse_categorical_crossentropy', 
						optimizer=Adam(lr=0.001),
						metrics=['accuracy'])
# discriminator.save_weights('./saved_model/model1119.h5')
discriminator.load_weights("./saved_model/model1119.h5")

# es_learn = EarlyStopping(monitor='val_acc', mode='max', 
# 						verbose=1, patience=10, restore_best_weights=True)
# discriminator.fit(x=X_train, y=y_train,
# 				callbacks=[es_learn], epochs=50, batch_size=128, validation_split=.2)

X, y = np.vstack((X_train, X_test)), np.hstack((y_train, y_test))

from sklearn.model_selection import KFold
kf = KFold(n_splits=10)

# ===================== performance for deep_taylor ===================
# mean_norm: 6.152(0.119), max_norm: 11.698(0.228); R_sqaure_test: 0.236(0.016)
# [10.298817, 11.444435, 11.963966, 12.210665, 11.862201, 12.417687, 11.985232, 10.597407, 11.525963, 12.678076]

# ===================== performance for gradient ===================
# mean_norm: 14.299(0.087), max_norm: 26.028(0.319); R_sqaure_test: 0.289(0.012)
# [25.423897, 27.173012, 26.974976, 25.435585, 24.46801, 26.40347, 25.071651, 27.639091, 26.581453, 25.11181 ]

# ===================== performance for lrp.z ===================
# mean_norm: 6.066(0.080), max_norm: 14.689(0.219); R_sqaure_test: 0.256(0.014)
# [13.922632, 14.834457, 14.890181, 16.168488, 13.913617, 15.439697, 14.788698, 14.381656, 14.680804, 13.869359]

# ===================== performance for deconvnet ===================
# mean_norm: 9.143(0.214), max_norm: 27.832(0.955); R_sqaure_test: 0.175(0.015)
# [21.651342, 24.710241, 28.045866, 30.828543, 32.101677, 31.107693, 28.460411, 27.637493, 25.652727, 28.126455]

# ===================== performance for pattern.net ===================
# mean_norm: 335.581(0.041), max_norm: 374.709(2.762); R_sqaure_test: 0.648(0.006)
# [379.23633, 375.27335, 349.87256, 377.40833, 374.8883 , 375.04938, 378.13184, 377.5188 , 375.09332, 384.6132 ]

R_sqaure_lst = []
k = 0
tau_range = [10.298817, 11.444435, 11.963966, 12.210665, 11.862201, 12.417687, 11.985232, 10.597407, 11.525963, 12.678076]
for tr, te in kf.split(X):
	X_train, y_train = X[tr], y[tr]
	X_test, y_test = X[te], y[te]
	train_loss_base, train_acc_base = discriminator.evaluate(X_train, y_train)
	test_loss_base, test_acc_base = discriminator.evaluate(X_test, y_test)

	print('Baseline: train_loss: %.3f; test_loss: %.3f' %(train_loss_base, test_loss_base))
	print('Baseline: train_acc: %.3f; test_acc: %.3f' %(train_acc_base, test_acc_base))

	print('===================== performance for proposed method ===================')
	tau_tmp = tau_range[k]
	k += 1
	detector = build_detector(img_shape=input_shape, lam=tau_tmp, type_='mask')
	discriminator_tmp = build_discriminator(img_shape=input_shape, labels=labels)
	discriminator_tmp.compile(loss='sparse_categorical_crossentropy', 
							optimizer=Adam(lr=0.001),
							metrics=['accuracy'])

	shiing = LocalGAN(input_shape=input_shape,
				labels=labels,
				discriminator=discriminator_tmp,
				detector=detector,
				optimizer=SGD(lr=1.),
				# optimizer=SGD(lr=0.001),
				task='classification')
	es_detect1 = ReduceLROnPlateau(monitor="loss", factor=0.2, min_lr=0.001, 
							verbose=1, patience=5, mode="min")
	es_detect2 = EarlyStopping(monitor='loss', mode='min', min_delta=.0001, 
							verbose=1, patience=10, restore_best_weights=True)

	shiing.discriminator.load_weights("./saved_model/model1119.h5")

	print('###'*20)
	print('#'*16+' '*5+'Train detector'+' '*5+'#'*16)
	print('###'*20)

	detect_tmp = shiing.combined.fit(x=X_train, y=y_train, callbacks=[es_detect1, es_detect2], 
									epochs=1000, batch_size=128)

	X_train_noise = shiing.detector.predict(X_train)
	X_test_noise = shiing.detector.predict(X_test)

	# train_loss, train_acc = shiing.discriminator.evaluate(X_train_noise, y_train)
	shiing_test_loss, shiing_test_acc = shiing.discriminator.evaluate(X_test_noise, y_test)
	print('tau: %.2f; test_loss: %.3f; test_acc: %.3f' %(tau_tmp, shiing_test_loss, shiing_test_acc))
	# shiing.R_square_train = 1. - train_loss_base / train_loss
	shiing.R_sqaure_test = 1. - test_loss_base / shiing_test_loss	
	print('tau: %.2f; R_sqaure_test: %.3f' %(tau_tmp, shiing.R_sqaure_test))
	R_sqaure_lst.append(shiing.R_sqaure_test)
print('proposed method: R_sqaure_test: %.3f(%.3f)' %(np.mean(R_sqaure_lst), np.std(R_sqaure_lst)/np.sqrt(10)))