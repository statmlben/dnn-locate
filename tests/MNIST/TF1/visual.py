## TF1.15; Keras2.2.4; H5<3.0.0

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
demo_ind = np.array([np.where(y_test==7)[0][0], np.where(y_test==9)[0][17]])

discriminator = build_discriminator(img_shape=input_shape, labels=labels)
discriminator.compile(loss='sparse_categorical_crossentropy', 
						optimizer=Adam(lr=0.001),
						metrics=['accuracy'])
# discriminator.save_weights('./saved_model/model1119.h5')
discriminator.load_weights("./saved_model/model1119.h5")

train_loss_base, train_acc_base = discriminator.evaluate(X_train, y_train)
test_loss_base, test_acc_base = discriminator.evaluate(X_test, y_test)

print('Baseline: train_loss: %.3f; test_loss: %.3f' %(train_loss_base, test_loss_base))
print('Baseline: train_acc: %.3f; test_acc: %.3f' %(train_acc_base, test_acc_base))


# 'deep_taylor', 'gradient', 'lrp.z', 'deconvnet', 'pattern.net'
import innvestigate
R_square_test_lst, X_test_R, X_test_noise_R = [], [], []
xai_dict = {'method': 'pattern.net', 'norm_max': [], 'norm_mean': [], 'R_sqaure': []}

method_name_lst = ['deep_taylor', 'gradient', 'lrp.z', 'deconvnet', 'pattern.net']
for method_name in method_name_lst:
	## apply analysis
	model = innvestigate.utils.model_wo_softmax(discriminator)
	analyzer = innvestigate.create_analyzer(method_name, model)
	## deep_taylor method
	if method_name == 'pattern.net':
		analyzer.fit(X_train, batch_size=64, verbose=1)
	X_test_diff = analyzer.analyze(X_test)
	X_test_diff /= np.max(np.abs(X_test_diff))
	X_test_noise = X_test - np.abs(X_test_diff)*X_test
	# plt.imshow(X_test_diff[0])
	# plt.show()
	norm_tmp = np.sum(np.abs(X_test_diff), (1,2,3))
	test_loss, test_acc = discriminator.evaluate(X_test_noise, y_test)
	print('===================== CV ===================')
	print('test_loss: %.3f; test_acc: %.3f' %(test_loss, test_acc))
	R_sqaure_test = 1. - test_loss_base / test_loss
	print('mean_norm: %.3f, max_norm: %.3f; R_sqaure_test: %.3f' %(norm_tmp.mean(), norm_tmp.max(), R_sqaure_test))
	R_square_test_lst.append(R_sqaure_test)	
	X_test_R.append(X_test[demo_ind])
	X_test_noise_R.append(X_test_diff[demo_ind])	
	# X_test_noise_R.append(X_test_noise[demo_ind])

R_square_test_lst, X_test_R, X_test_noise_R = np.array(R_square_test_lst), np.array(X_test_R), np.array(X_test_noise_R)
show_samples(R_square_test_lst, X_test_R, X_test_noise_R, method_name=method_name_lst, threshold=[1e-2]*5, mix='diff')


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
