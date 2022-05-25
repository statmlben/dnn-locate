from __future__ import print_function, division
import tensorflow as tf
from keras.datasets import mnist
# from tensorflow.python.keras import backend
from tensorflow.keras import backend
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add, Multiply, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import initializers
import numpy as np
from tensorflow.keras import Model, layers,Sequential,regularizers


initializer = initializers.glorot_uniform(seed=0)
# initializer = initializers.Ones()

def TRelu(x):
    return backend.relu(x, max_value=1.)

def build_detector(img_shape, lam, detector_backend=None, type_='mask'):

    if detector_backend == None:
        ## define detector backend with pre-trained VGG16 network
        detector_backend = tf.keras.applications.vgg16.VGG16(
            include_top=False, weights=None, input_tensor=None,
            input_shape=img_shape, pooling=None, classes=1000,
            classifier_activation='relu'
            )

    img = Input(shape=img_shape)
    backend_out = detector_backend(img)

    dense_out = Dense(np.prod(img_shape), 
                activation ='softmax',
                kernel_initializer=initializer,
                # activity_regularizer=tf.keras.regularizers.l1(lam),
                # activity_regularizer = entropy_reg(self.lam),
                bias_initializer=initializer)(backend_out)
    attent_out = Reshape(img_shape)(dense_out)
    attent_out = lam*attent_out
    attent_out = Activation(TRelu)(attent_out)

    # mask img
    if type_ == 'mask':
        atten_img = Multiply()([attent_out, img])
        detect_img = Add()([img, -atten_img])
        # img_noise = mask_img
    else:
        detect_img = Add()([img, attent_out])

    return Model(img, detect_img)


# def build_detector(img_shape, detector_backend, lam, type_='mask'):

# 	model = Sequential()
# 	model.add(Conv2D(32, (2,2),
# 		padding="same",
# 		input_shape=img_shape,
# 		kernel_initializer=initializer,
# 		bias_initializer=initializer))
# 	model.add(Flatten())
# 	model.add(Dense(128, activation='relu', 
# 		kernel_initializer=initializer, 
# 		bias_initializer=initializer))
# 	model.add(Dense(128, activation='relu',
# 		kernel_initializer=initializer, 
# 		bias_initializer=initializer))
    
#     if type_ == 'mask':
# 		model.add(Dense(np.prod(img_shape), 
# 			# activation ='sigmoid',
# 			activation ='softmax',
# 			# activity_regularizer=tf.keras.regularizers.l1(lam),
# 			# activity_regularizer=entropy_reg(self.lam),
# 			kernel_initializer=initializer,
# 			bias_initializer=initializer))
# 		# model.add(ReLU(max_value=1))
# 	else:
# 		model.add(Dense(np.prod(img_shape), 
# 				activation ='tanh',
# 				kernel_initializer=initializer,
# 				# activity_regularizer=tf.keras.regularizers.l1(lam),
# 				# activity_regularizer = entropy_reg(self.lam),
# 				bias_initializer=initializer))

# 	# model.add(ReLU(threshold=.1))
# 	model.add(Reshape(img_shape))

# 	img = Input(shape=img_shape)
# 	noise = model(img)
# 	noise = lam*noise
# 	noise = Activation(TRelu)(noise)

# 	# mask img
# 	if type_ == 'mask':
# 		mask_img = Multiply()([noise, img])
# 		img_noise = Add()([img, -mask_img])
# 		# img_noise = mask_img
# 	else:
# 		img_noise = Add()([img, noise])
# 	# img = model(noise)
# 	# model.summary()

# 	return Model(img, img_noise)

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
    # model.add(Dense(labels, activation='softmax', name='output_layer',
    # 	kernel_initializer=initializer,
    # 	kernel_regularizer=tf.keras.regularizers.l1(0.001),
    # 	bias_regularizer=tf.keras.regularizers.l1(0.001),
    # 	bias_initializer=initializer))
    model.add(Dense(labels, activation='softmax',
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l1(0.001),
        bias_regularizer=tf.keras.regularizers.l1(0.001),
        bias_initializer=initializer))
    # model.summary()
    return model


def build_discriminator_gap(img_shape, labels):
    model = Sequential()
    model.add(Conv2D(64, (2, 2),
            activation='relu',
            # padding="same",
            kernel_initializer=initializer,
            bias_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l1(0.001),
            bias_regularizer=tf.keras.regularizers.l1(0.001),
            input_shape=img_shape))
    model.add(Conv2D(64, (2, 2),
        name='last_conv',
        activation='relu',
        # padding="same",
        kernel_initializer=initializer,
        bias_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l1(0.001),
        bias_regularizer=tf.keras.regularizers.l1(0.001),
        input_shape=img_shape))
    model.add(MaxPooling2D((2, 2)))	
    model.add(GlobalAveragePooling2D())
    # model.add(Flatten())
    # model.add(Dense(100, activation='relu',
    # 	kernel_regularizer=tf.keras.regularizers.l1(0.001),
    # 	bias_regularizer=tf.keras.regularizers.l1(0.001),
    # 	kernel_initializer=initializer))
    model.add(Dense(labels, activation='softmax', name='output_layer',
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l1(0.001),
        bias_regularizer=tf.keras.regularizers.l1(0.001),
        bias_initializer=initializer))
    # model.summary()
    # img = Input(shape=img_shape)
    # prob = model(img)
    return model

def CAENet2D(input_shape):
    detector_backend = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(
                filters=32, kernel_size=5, padding="same", strides=1, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv2D(
                filters=16, kernel_size=5, padding="same", strides=1, activation="relu"
            ),
            layers.Conv2DTranspose(
                filters=16, kernel_size=5, padding="same", strides=1, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv2DTranspose(
                filters=32, kernel_size=5, padding="same", strides=1, activation="relu"
            ),
            layers.Conv2DTranspose(filters=1, kernel_size=5, padding="same"),
        ]
    )
    return detector_backend