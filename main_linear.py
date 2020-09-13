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
# plot for 1d case
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#sphx-glr-auto-examples-linear-model-plot-ransac-py



n_samples, input_shape, labels = 5000, 100, 1

noise = 10
X = np.random.randn(n_samples, input_shape)
beta_true = np.random.randn(input_shape)
beta_true[int(input_shape/10):] = 0.
y = X.dot(beta_true) + noise*np.random.randn(n_samples)

# X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=input_shape,
#                                       	n_informative=int(input_shape/10)+1, noise=10.,
#                                       	coef=True, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## define models

discriminator = build_discriminator(input_dim=input_shape, 
									labels=labels, 
									lam=0.0001)

discriminator.compile(loss='MSE', optimizer=Adam(lr=0.1))
# discriminator.load_weights("model_noise.h5")


es_learn = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)

# fit discriminator
print('###'*20)
print('###'*5+' '*6+'Train for learner'+' '*5+'###'*5)
print('###'*20)

learn_tmp = discriminator.fit(x=X_train, y=y_train, callbacks=[es_learn], epochs=1000, 
									batch_size=128, validation_split=.2)

train_loss_base = discriminator.evaluate(X_train, y_train)
test_loss_base = discriminator.evaluate(X_test, y_test)

## define framework
# lam_range = [.2]
lam_range = np.arange(.2, 10.0, .2)
lam_range = lam_range.astype('float32')
R_square_train_lst, R_square_test_lst = [], []

for lam in lam_range:
	PSGD = ProximalGradientDescentOptimizer(learning_rate = 0.01,
									l1_regularization_strength = lam)

	detector = build_detector(input_dim=input_shape,
								regularizer=tf.keras.regularizers.L1(0.), 
								type_='noise')

	shiing = LocalGAN(input_shape=input_shape, 
						labels=labels,
						discriminator=discriminator, 
						detector=detector,
						optimizer=PSGD,
						# optimizer=SGD(lr=0.001),
						task='regression')

	es_detect = EarlyStopping(monitor='loss', mode='min', min_delta=.0001, verbose=1, patience=1000, restore_best_weights=True)

	# shiing.discriminator.save_weights("model_01.h5")
	# shiing.discriminator.load_weights("model_test.h5")
	# shiing.discriminator.load_weights("model_01.h5")
	# shiing.discriminator.load_weights("model_noise.h5")

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
	# validation_split=.2

	## show the results
	X_train_noise = shiing.detector.predict(X_train)
	X_test_noise = shiing.detector.predict(X_test)

	train_loss = shiing.discriminator.evaluate(X_train_noise, y_train)
	test_loss = shiing.discriminator.evaluate(X_test_noise, y_test)

	print('Baseline: train_loss: %.3f; test_loss: %.3f' %(train_loss_base, test_loss_base))
	# print('Baseline: train_acc: %.3f; test_acc: %.3f' %(train_acc_base, test_acc_base))
	print('lam: %.3f; train_loss: %.3f; test_loss: %.3f' %(lam, train_loss, test_loss))
	# print('lam: %.5f; train_acc: %.3f; test_acc: %.3f' %(lam, train_acc, test_acc))

	shiing.R_square_train = 1. - train_loss_base / train_loss
	shiing.R_sqaure_test = 1. - test_loss_base / test_loss
	print('lam: %.3f; R_square_train: %.3f; R_sqaure_test: %.3f' 
		%(lam, shiing.R_square_train, shiing.R_sqaure_test))
	print('true parameter: %s' %beta_true)
	print('delta: %s' %(X_train_noise[0] - X_train[0]))
	R_square_train_lst.append(shiing.R_square_train)
	R_square_test_lst.append(shiing.R_sqaure_test)

# print(R_square_train_lst)
# print(R_square_test_lst)

# R_square_train_lst = [0.795597143873443, 0.7901153606659239, 0.7846094000733868, 0.7789560364211161, 0.7714482547160242, 0.7629838740349997, 0.7536717177839927, 0.7426291531312593, 0.7302834533541153, 0.7175523748776889, 0.7041529203703243, 0.68509710793004, 0.6701807062595773, 0.653334139669022, 0.6381600634320178, 0.6180961624126166, 0.6030289722959752, 0.553438154618743, 0.31199087160303973, 0.29564680161896484, 0.2836821038485833, 0.2744524557911976, 0.2670626702429504, 0.2500481793877031, 0.2408908893396844, 0.23602767358099952, 0.20995864866136282, 0.18273674568854936, 0.17086601856155814, 0.17766644452546188, 0.1503873404812066, 0.15758883063166018, 0.149453429595371, 0.1448757531079028, 0.0, 0.0013313429568163615, 0.0, 0.0, 0.0, -2.187558242749965e-05, -0.0002391828792451811, 0.0, -0.00024613574386278536, 0.0, 0.0]
# R_square_test_lst = [0.7620997058251329, 0.7557842104067081, 0.7494478321334521, 0.7429492546116188, 0.7343313413981409, 0.7246321646957352, 0.713982756618035, 0.7013846159078267, 0.6873394423303818, 0.6729014618022003, 0.6577573366330365, 0.6363153497586312, 0.6196114241722354, 0.6008342892915457, 0.5840041412097283, 0.5618745346470839, 0.545351903052025, 0.49157873158821785, 0.24534306568425002, 0.229766317018686, 0.21846560692069905, 0.20980900010261438, 0.2029169382173075, 0.18718423175563137, 0.17879704191082546, 0.17436654290660847, 0.15090640105947273, 0.12696527680311476, 0.1167185120638844, 0.12257370481514773, 0.09934313178587806, 0.10540825991882508, 0.0985604549633633, 0.09473636896275117, 0.0, -0.0014803026897347582, 0.0, 0.0, 0.0, 0.0035592061937771424, 0.0010226265400620216, 0.0, 0.0021385960174660212, 0.0, 0.0]

# import pandas as pd
# import seaborn as sns
# R_train = pd.DataFrame({'lam': lam_range, 'R_square': R_square_train_lst, 'Type': ['R_square_train']*45})
# R_test = pd.DataFrame({'lam': lam_range, 'R_square': R_square_test_lst, 'Type': ['R_square_test']*45})
# df = pd.concat([R_train, R_test])

# sns.set()
# sns.lineplot(data=df, x="lam", y="R_square", color='k', markers=True, alpha=.7, style='Type', lw=2.)
# plt.show()