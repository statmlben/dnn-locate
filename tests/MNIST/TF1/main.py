from keras.datasets import mnist
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from models.cnn_models import build_detector, build_discriminator
from models.cnn_models_v2 import build_detector, build_discriminator

from dnn_locate import LocalGAN
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

demo_ind = np.array([np.where(y_test==7)[0][2], np.where(y_test==9)[0][2]])
# demo_ind = np.array([np.where(y_test==7)[0][10], np.where(y_test==9)[0][31]])
# plt.imshow(X_test[demo_ind[0]])
# plt.show()
# plt.imshow(X_test[demo_ind[1]])
# plt.show()
## define models
# lam_range = [6, 8, 10, 12, 14, 15, 16, 18, 20]
lam_range = [6, 8, 10, 12, 14, 16, 18, 20, 22]
R_square_train_lst, R_square_test_lst, norm_lst, norm_test_lst, X_test_R, X_test_noise_R = [], [], [], [], [], []

for lam in lam_range:
	detector = build_detector(img_shape=input_shape, lam=lam, type_='mask')

	discriminator = build_discriminator(img_shape=input_shape, labels=labels)
	discriminator.compile(loss='sparse_categorical_crossentropy', 
							optimizer=Adam(lr=0.001),
							metrics=['accuracy'])

	## define framework
	shiing = LocalGAN(input_shape=input_shape,
					labels=labels,
					discriminator=discriminator,
					detector=detector,
					optimizer=SGD(lr=1.),
					# optimizer=SGD(lr=0.001),
					task='classification')
	
	es_detect1 = ReduceLROnPlateau(monitor="loss", factor=0.382, min_lr=0.0001, 
							verbose=1, patience=4, mode="min")
	es_detect2 = EarlyStopping(monitor='loss', mode='min', min_delta=.00001, 
							verbose=1, patience=10, restore_best_weights=False)

	es_learn = EarlyStopping(monitor='val_accuracy', mode='max', 
							verbose=1, patience=10, restore_best_weights=True)

	print('###'*20)
	print('###'*5+' '*6+'Load learner'+' '*5+'###'*5)
	print('###'*20)

	# learn_tmp = shiing.discriminator.fit(x=X_train, y=y_train, callbacks=[es_learn], epochs=50, batch_size=128, validation_split=.2)
	# shiing.discriminator.save_weights("model1107.h5")
	shiing.discriminator.load_weights("./saved_model/model1107.h5")
	train_loss_base, train_acc_base = shiing.discriminator.evaluate(X_train, y_train)
	test_loss_base, test_acc_base = shiing.discriminator.evaluate(X_test, y_test)

	print('Baseline: train_loss: %.3f; test_loss: %.3f' %(train_loss_base, test_loss_base))
	print('Baseline: train_acc: %.3f; test_acc: %.3f' %(train_acc_base, test_acc_base))


	print('###'*20)
	print('#'*16+' '*5+'Train detector'+' '*5+'#'*16)
	print('###'*20)

	detect_tmp = shiing.combined.fit(x=X_train, y=y_train, callbacks=[es_detect1, es_detect2], 
									epochs=1000, batch_size=128)
	# validation_split=.2

	## show the results
	X_train_noise = shiing.detector.predict(X_train)
	X_test_noise = shiing.detector.predict(X_test)

	if method == 'mask':
		X_diff = np.nan_to_num( (X_train_noise - X_train) / (X_train+1e-8) )
		X_diff_test = np.nan_to_num( (X_test_noise - X_test) / (X_test+1e-8) )
	elif method == 'noise':
		X_diff = X_train_noise[0,:,:,0] - X_train[0,:,:,0]
		X_diff_test = X_test_noise[0,:,:,0] - X_test[0,:,:,0]

	norm_tmp = np.sum(np.abs(X_diff))/len(X_diff)
	norm_tmp_test = np.sum(np.abs(X_diff_test))/len(X_diff_test)

	train_loss, train_acc = shiing.discriminator.evaluate(X_train_noise, y_train)
	test_loss, test_acc = shiing.discriminator.evaluate(X_test_noise, y_test)
	print('lam: %.2f; train_loss: %.3f; test_loss: %.3f' %(lam, train_loss, test_loss))
	print('lam: %.2f; train_acc: %.3f; test_acc: %.3f' %(lam, train_acc, test_acc))

	shiing.R_square_train = 1. - train_loss_base / train_loss
	shiing.R_sqaure_test = 1. - test_loss_base / test_loss
	print('lam: %.2f; diff_norm: %.3f; R_square_train: %.3f; R_sqaure_test: %.3f' 
		%(lam, norm_tmp, shiing.R_square_train, shiing.R_sqaure_test))
	R_square_train_lst.append(shiing.R_square_train)
	R_square_test_lst.append(shiing.R_sqaure_test)
	norm_lst.append(norm_tmp)
	norm_test_lst.append(norm_tmp_test)
	X_test_R.append(X_test[demo_ind])
	X_test_noise_R.append(X_test_noise[demo_ind])

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