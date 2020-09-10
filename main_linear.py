import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from dnn_locate import LocalGAN
from models.linear_models import build_detector, build_discriminator
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD

# plot for 1d case
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#sphx-glr-auto-examples-linear-model-plot-ransac-py

n_samples, input_shape, labels = 1000, 1, 1

X_train, y_train, coef = datasets.make_regression(n_samples=n_samples, n_features=input_shape,
                                      				n_informative=int(input_shape/10)+1, noise=10.,
                                      				coef=True, random_state=1)

## define models 
detector = build_detector(input_dim=input_shape, lam=0.1, type_='noise')
discriminator = build_discriminator(input_dim=input_shape, labels=labels, lam=0.0)
discriminator.compile(loss='MSE', optimizer=Adam(lr=0.01))
# discriminator.load_weights("model_noise.h5")

## define framework
elk = LocalGAN(input_shape=input_shape, labels=labels, discriminator=discriminator, detector=detector)
es_detect = EarlyStopping(monitor='loss', mode='min', min_delta=.0001, verbose=1, patience=3, restore_best_weights=True)
es_learn = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)

print('###'*20)
print('###'*5+' '*6+'Train for learner'+' '*5+'###'*5)
print('###'*20)

learn_tmp = elk.discriminator.fit(x=X_train, y=y_train, callbacks=[es_learn], epochs=1000, batch_size=128, validation_split=.2)
# elk.discriminator.save_weights("model_01.h5")
# elk.discriminator.load_weights("model_test.h5")
# elk.discriminator.load_weights("model_01.h5")
# elk.discriminator.load_weights("model_noise.h5")
train_loss_base = elk.discriminator.evaluate(X_train, y_train)
# test_loss_base = elk.discriminator.evaluate(X_test, y_test)

# lw = 2
y_pred = elk.discriminator.predict(X_train)
plt.scatter(X_train, y_pred, color='yellowgreen', marker='.')
plt.scatter(X_train, y_train, color='red', marker='.')
# plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
# plt.legend(loc='lower right')
# plt.xlabel("Input")
# plt.ylabel("Response")
plt.show()

print('Baseline: train_loss: %.3f; test_loss: %.3f' %(train_loss_base, test_loss_base))
# print('Baseline: train_acc: %.3f; test_acc: %.3f' %(train_acc_base, test_acc_base))

## mark here Sep 9th

print('###'*20)
print('#'*16+' '*5+'Train for detector'+' '*5+'#'*16)
print('###'*20)

detect_tmp = elk.combined.fit(x=X_train, y=y_train, callbacks=[es_detect], 
								epochs=300, batch_size=128)
# validation_split=.2

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