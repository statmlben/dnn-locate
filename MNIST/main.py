from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from dnn_locate.loc_model import loc_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW, SGDW
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model, layers,Sequential,regularizers

np.random.seed(3)
tf.random.set_seed(3)

input_shape, labels = (28, 28, 1), 10

## load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

X_train = X_train / 255.
X_test = X_test / 255.

# ind_set = np.array([i for i in range(len(y_train)) if y_train[i] in [7, 9]])
# ind_set_test = np.array([i for i in range(len(y_test)) if y_test[i] in [7, 9]])

# X_train, y_train = X_train[ind_set], y_train[ind_set]
# X_test, y_test = X_test[ind_set_test], y_test[ind_set_test]

X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)


# specify the arguments
rotation_range_val = 30
width_shift_val = 0.25
height_shift_val = 0.25
shear_range_val = 45
zoom_range_val = [0.5,1.5]

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range = rotation_range_val, 
                            width_shift_range = width_shift_val, 
                            height_shift_range = height_shift_val,
                            zoom_range=zoom_range_val)
datagen.fit(X_train)

## define models
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Add, Multiply
from keras.layers import Activation, MaxPooling2D, Conv2D

initializer = initializers.glorot_uniform(seed=0)

## define the detector before TRELU activation
# detector_backend = Sequential()
# detector_backend.add(Conv2D(32, (2,2),
# 	padding="same",
# 	input_shape=input_shape,
# 	kernel_initializer=initializer, 
# 	bias_initializer=initializer))
# detector_backend.add(Flatten())
# detector_backend.add(Dense(128, activation='relu', 
# 	kernel_initializer=initializer, 
# 	bias_initializer=initializer))
# detector_backend.add(Dense(128, activation='relu',
# 	kernel_initializer=initializer, 
# 	bias_initializer=initializer))
# detector_backend.add(Dense(np.prod(input_shape), 
# 	activation ='softmax',
# 	kernel_initializer=initializer,
# 	bias_initializer=initializer))
# detector_backend.add(Reshape(input_shape))
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

## define discriminator
discriminator = Sequential()
discriminator.add(Conv2D(64, (3, 3),
		activation='relu',
		kernel_initializer=initializer,
		bias_initializer=initializer,
		kernel_regularizer=tf.keras.regularizers.l1(0.001),
		bias_regularizer=tf.keras.regularizers.l1(0.001),
		input_shape=input_shape))
discriminator.add(Conv2D(32, (3, 3),
		activation='relu', name='last_conv',
		kernel_initializer=initializer,
		bias_initializer=initializer,
		kernel_regularizer=tf.keras.regularizers.l1(0.001),
		bias_regularizer=tf.keras.regularizers.l1(0.001),
		input_shape=input_shape))
discriminator.add(MaxPooling2D((2, 2)))
discriminator.add(Flatten())
discriminator.add(Dense(512, activation='relu',
	kernel_regularizer=tf.keras.regularizers.l1(0.001),
	bias_regularizer=tf.keras.regularizers.l1(0.001),
	kernel_initializer=initializer))
discriminator.add(Dense(labels, activation='softmax',
	kernel_initializer=initializer,
	kernel_regularizer=tf.keras.regularizers.l1(0.001),
	bias_regularizer=tf.keras.regularizers.l1(0.001),
	bias_initializer=initializer))
discriminator.compile(loss='categorical_crossentropy', 
						optimizer='Adam',
						metrics=['accuracy'])

es_detect1 = ReduceLROnPlateau(monitor="loss", factor=0.382, min_lr=1e-6,
                    verbose=1, patience=3, mode="min")
es_detect2 = EarlyStopping(monitor='loss', mode='min', min_delta=.0001, 
                        verbose=1, patience=20, restore_best_weights=True)
fit_params={'callbacks': [es_detect1, es_detect2],
            'epochs': 200, 'batch_size': 128}

discriminator.fit(
        datagen.flow(
        X_train, y_train), **fit_params)

## define framework
tau_range = [20.]
cue = loc_model(input_shape=input_shape,
                detector_backend=detector_backend,
                discriminator=discriminator,
                target_r_square='auto',
                r_metric='acc',
                # r_metric='loss',
                tau_range=tau_range)

cue.fit(X_train=X_train, y_train=y_train, datagen=datagen,
        optimizer=SGDW(learning_rate=.1, weight_decay=.0001, momentum=.9), fit_params=fit_params)

for X_demo, Y_demo in datagen.flow(X_train, y_train[:,np.newaxis], batch_size=6):
    break

X_demo_detect = cue.detector.predict(X_demo)
X_demo_hl =  - (X_demo_detect - X_demo) / (X_demo + 1e-5)


fig, ax = plt.subplots(1, 6)
for i in range(len(X_demo)):
    X_tmp, X_detect_tmp, X_hl_tmp = X_demo[i], X_demo_detect[i], X_demo_hl[i]
    ax[i].imshow(X_tmp, cmap='gray_r')
    ax[i].imshow(X_hl_tmp, cmap="OrRd", alpha=0.3)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.show()