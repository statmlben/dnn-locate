from keras.datasets import mnist
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from models.cnn_models import build_detector, build_discriminator
from models.cnn_models_v2 import build_detector, build_discriminator

import matplotlib.pyplot as plt
import sys
import numpy as np
from dnn_locate import LocalGAN
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GaussianNoise
from EDA import show_samples, R_sqaure_path, show_diff_samples

input_shape, labels = (28, 28, 1), 10
sample = GaussianNoise(0.2)

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
## define models
# lam_range = [5, 6, 8, 10, 12, 14, 15, 16, 18, 20]
tau = 20

detector = build_detector(img_shape=input_shape, lam=tau, type_='mask')

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

es_detect1 = ReduceLROnPlateau(monitor="loss", factor=0.618, min_lr=0.0001, 
						verbose=1, patience=4, mode="min")
es_detect2 = EarlyStopping(monitor='loss', mode='min', min_delta=.00001, 
						verbose=1, patience=15, restore_best_weights=False)

es_learn = EarlyStopping(monitor='val_accuracy', mode='max', 
						verbose=1, patience=10, restore_best_weights=True)

print('###'*20)
print('###'*5+' '*6+'Load learner'+' '*5+'###'*5)
print('###'*20)

# learn_tmp = shiing.discriminator.fit(x=X_train, y=y_train, callbacks=[es_learn], epochs=50, batch_size=128, validation_split=.2)
# shiing.discriminator.save_weights("model1107.h5")
shiing.discriminator.load_weights("model1107.h5")
train_loss_base, train_acc_base = shiing.discriminator.evaluate(X_train, y_train)
test_loss_base, test_acc_base = shiing.discriminator.evaluate(X_test, y_test)

print('Baseline: train_loss: %.3f; test_loss: %.3f' %(train_loss_base, test_loss_base))
print('Baseline: train_acc: %.3f; test_acc: %.3f' %(train_acc_base, test_acc_base))


print('###'*20)
print('#'*16+' '*5+'Train detector'+' '*5+'#'*16)
print('###'*20)

detect_tmp = shiing.combined.fit(x=X_train, y=y_train, callbacks=[es_detect1, es_detect2], 
								epochs=500, batch_size=128)
# validation_split=.2

## show the results
from sklearn.utils import resample
R_square_train_lst, R_square_test_lst, norm_lst, norm_test_lst = [], [], [], []

B_samples = 500
for b in range(B_samples):
	X_test_boot = resample(X_test, replace=True, n_samples=len(X_test), random_state=b)
	X_train_boot = resample(X_train, replace=True, n_samples=len(X_train), random_state=b)
	
	X_train_noise = shiing.detector.predict(X_train_boot)
	X_test_noise = shiing.detector.predict(X_test_boot)

	if method == 'mask':
		X_diff = np.nan_to_num( (X_train_noise - X_train_boot) / (X_train_boot+1e-8) )
		X_diff_test = np.nan_to_num( (X_test_noise - X_test_boot) / (X_test_boot+1e-8) )
	elif method == 'noise':
		X_diff = X_train_noise[0,:,:,0] - X_train_boot[0,:,:,0]
		X_diff_test = X_test_noise[0,:,:,0] - X_test_boot[0,:,:,0]

	norm_tmp = np.sum(np.abs(X_diff))/len(X_diff)
	norm_tmp_test = np.sum(np.abs(X_diff_test))/len(X_diff_test)

	train_loss, train_acc = shiing.discriminator.evaluate(X_train_noise, y_train)
	test_loss, test_acc = shiing.discriminator.evaluate(X_test_noise, y_test)
	print('tau: %.2f; train_loss: %.3f; test_loss: %.3f' %(tau, train_loss, test_loss))
	print('tau: %.2f; train_acc: %.3f; test_acc: %.3f' %(tau, train_acc, test_acc))

	shiing.R_square_train = 1. - train_loss_base / train_loss
	shiing.R_sqaure_test = 1. - test_loss_base / test_loss
	print('tau: %.2f; diff_norm: %.3f; R_square_train: %.3f; R_sqaure_test: %.3f' 
		%(tau, norm_tmp, shiing.R_square_train, shiing.R_sqaure_test))
	R_square_train_lst.append(shiing.R_square_train)
	R_square_test_lst.append(shiing.R_sqaure_test)
	norm_lst.append(norm_tmp)
	norm_test_lst.append(norm_tmp_test)

print('R_square_train: (2.5%% quantile: %.4f; 97.5%% quantile: %.4f)' 
	%(np.quantile(R_square_train_lst, q=.025), np.quantile(R_square_train_lst, q=.975)))

print('R_square_test: (2.5%% quantile: %.4f; 97.5%% quantile: %.4f)' 
	%(np.quantile(R_square_test_lst, q=.025), np.quantile(R_square_test_lst, q=.975)))

R_sqaure_train_dict = {'data': ['R_square_train']*B_samples, 'R_sqaure': R_square_train_lst}
R_sqaure_test_dict = {'data': ['R_square_test']*B_samples, 'R_sqaure': R_square_test_lst}
R_sqaure_dict = {key:np.append(R_sqaure_train_dict[key], R_sqaure_test_dict[key]) for key in R_sqaure_train_dict}

import seaborn as sns
sns.set()
ax = sns.boxplot(x="R_sqaure", y='data', data=R_sqaure_dict, color='gray', whis=[2.5, 97.5], width=.6)
ax = sns.stripplot(x="R_sqaure", y="data", data=R_sqaure_dict, color=".35", size=5, linewidth=0)
plt.show()


# X_test_demo, X_test_noise_demo = [], []
# for i in range(5,14):
# 	demo_ind = np.array([np.where(y_test==7)[0][i], np.where(y_test==9)[0][i]])
# 	X_test_demo.append(X_test[demo_ind])
# 	X_test_noise_demo.append(X_test_noise[demo_ind])
# X_test_demo, X_test_noise_demo = np.array(X_test_demo), np.array(X_test_noise_demo)
# show_diff_samples(X_test=X_test_demo, X_test_noise=X_test_noise_demo)