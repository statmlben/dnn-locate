import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
from dnn_locate import LocalGAN
from models.linear_models import build_detector, build_discriminator
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
from tensorflow.compat.v1.train import ProximalGradientDescentOptimizer
from sklearn.model_selection import train_test_split
from keras import initializers

# plot for 1d case
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#sphx-glr-auto-examples-linear-model-plot-ransac-py

n_samples, input_shape, labels = 5000, 100, 1

noise = 10
X = np.random.randn(n_samples, input_shape)
# beta_true = np.random.randn(input_shape)
beta_true = np.zeros(input_shape)
beta_true[:int(input_shape/10)] = np.arange(1., 11, 1.)
y = X.dot(beta_true) + noise*np.random.randn(n_samples)

# X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=input_shape,
#                                       	n_informative=int(input_shape/10)+1, noise=10.,
#                                       	coef=True, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## define models

discriminator = build_discriminator(input_dim=input_shape, 
									labels=labels, 
									lam=0.001)

discriminator.compile(loss='MSE', optimizer=Adam(lr=0.001))
# discriminator.load_weights("model_noise.h5")


es_learn = EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
						patience=1000, restore_best_weights=False)

# fit discriminator
print('###'*20)
print('###'*5+' '*6+'Train for learner'+' '*5+'###'*5)
print('###'*20)

# learn_tmp = discriminator.fit(x=X_train, y=y_train, callbacks=[es_learn], epochs=1, 
# 									batch_size=128, validation_split=.2)

discriminator.set_weights(beta_true.reshape(1, input_shape, 1)) 
train_loss_base = discriminator.evaluate(X_train, y_train)
test_loss_base = discriminator.evaluate(X_test, y_test)

## define framework
# lam_range = [.5]
lam_range = np.arange(200., 260., 10.)
# lam_range = lam_range.astype('float32')
R_square_train_lst, R_square_test_lst = [], []
delta_lst = []
for lam in lam_range:
	loss_seed, delta_seed, train_loss_seed, test_loss_seed = [], [], [], []
	for l in range(1):
		# PSGD = ProximalGradientDescentOptimizer(learning_rate = 0.001,
		# 								l1_regularization_strength = lam)
		if l == 0:
			initializer_tmp = initializers.Ones()
		else:
			initializer_tmp = initializers.Zeros()

		detector = build_detector(input_dim=input_shape,
									initializer=initializer_tmp,
									regularizer=tf.keras.regularizers.L1(lam), 
									type_='noise')

		shiing = LocalGAN(input_shape=input_shape,
							labels=labels,
							discriminator=discriminator, 
							detector=detector,
							# optimizer=PSGD,
							optimizer=Adam(lr=0.001),
							task='regression')

		es_detect = EarlyStopping(monitor='loss', mode='min', min_delta=.0001, 
								verbose=1, patience=100, restore_best_weights=False)
		
		# lw = 2
		# y_pred = shiing.discriminator.predict(X_train)
		# plt.scatter(X_train, y_pred, color='yellowgreen', marker='.')
		# plt.scatter(X_train, y_train, color='red', marker='.')
		# # plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
		# # plt.legend(loc='lower right')
		# # plt.xlabel("Input")
		# # plt.ylabel("Response")
		# plt.show()

		print('###'*20)
		print('#'*16+' '*5+'Train for detector'+' '*5+'#'*16)
		print('###'*20)

		detect_tmp = shiing.combined.fit(x=X_train, y=y_train, callbacks=[es_detect], 
										epochs=10000, batch_size=128)
		loss_seed.append(detect_tmp.history['loss'][-1])
		## show the results
		X_train_noise = shiing.detector.predict(X_train)
		X_test_noise = shiing.detector.predict(X_test)

		delta_tmp = X_train_noise[0] - X_train[0]
		train_loss_tmp = shiing.discriminator.evaluate(X_train_noise, y_train)
		test_loss_tmp = shiing.discriminator.evaluate(X_test_noise, y_test)

		delta_seed.append(delta_tmp)
		train_loss_seed.append(train_loss_tmp), 
		test_loss_seed.append(test_loss_tmp)

	delta_tmp = delta_seed[np.argmin(loss_seed)]
	train_loss = train_loss_seed[np.argmin(loss_seed)]
	test_loss = test_loss_seed[np.argmin(loss_seed)]

	print('Baseline: train_loss: %.3f; test_loss: %.3f' %(train_loss_base, test_loss_base))
	# print('Baseline: train_acc: %.3f; test_acc: %.3f' %(train_acc_base, test_acc_base))
	print('lam: %.3f; train_loss: %.3f; test_loss: %.3f' %(lam, train_loss, test_loss))
	# print('lam: %.5f; train_acc: %.3f; test_acc: %.3f' %(lam, train_acc, test_acc))

	shiing.R_square_train = 1. - train_loss_base / train_loss
	shiing.R_sqaure_test = 1. - test_loss_base / test_loss
	
	print('lam: %.3f; R_square_train: %.3f; R_sqaure_test: %.3f' 
		%(lam, shiing.R_square_train, shiing.R_sqaure_test))
	print('true parameter: %s' %beta_true[:10])
	print('delta: %s' %delta_tmp[:10])
	R_square_train_lst.append(shiing.R_square_train)
	R_square_test_lst.append(shiing.R_sqaure_test)
	delta_lst.append(delta_tmp)

print(R_square_train_lst)
print(R_square_test_lst)
delta_lst = np.abs(delta_lst)
import pandas as pd
import seaborn as sns
sns.set()
sns.heatmap(delta_lst)
plt.show()

R_train = pd.DataFrame({'lam': lam_range, 'R_square': R_square_train_lst, 'Type': ['R_square_train']*len(lam_range)})
R_test = pd.DataFrame({'lam': lam_range, 'R_square': R_square_test_lst, 'Type': ['R_square_test']*len(lam_range)})
df = pd.concat([R_train, R_test])
sns.lineplot(data=df, x="lam", y="R_square", color='k', markers=True, alpha=.7, style='Type', lw=2.)
plt.show()