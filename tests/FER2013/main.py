from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
from dnn_locate import loc_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import initializers
from tensorflow.keras.utils import to_categorical
from tensorflow_addons.optimizers import AdamW, SGDW

np.random.seed(3)
tf.random.set_seed(3)


x_train = np.load('../valid_image_array.npy')
y_train = np.load('../valid_label.npy')
x_train /= 255.
x_train = x_train.astype('float32')
y_train = to_categorical(y_train)

input_shape, labels = (48, 48, 1), 7

from vgg16 import VGG16Model

## define discriminator and detector_backend

discriminator = VGG16Model()

discriminator.compile(loss='categorical_crossentropy',
                        optimizer=SGDW(learning_rate=.01, weight_decay=.001, momentum=.9),
                        metrics=['accuracy'])
discriminator.load_weights('./saved_model/vgg16.h5')


## define the backend detector before TRELU activation
from autoencoder import AE, CAE
detector_backend = AE(input_shape, c_hid=32, latent_dim=1000)


# ## define framework
es_detect1 = ReduceLROnPlateau(monitor="loss", factor=0.382, min_lr=1e-6,
                    verbose=1, patience=3, mode="min")
es_detect2 = EarlyStopping(monitor='loss', mode='min', min_delta=.0001, 
                        verbose=1, patience=10, restore_best_weights=False)
fit_params={'callbacks': [es_detect1, es_detect2],
            'epochs': 200, 'batch_size': 128}

# detector_backend.compile(optimizer=SGD(lr=1.), loss='binary_crossentropy')
# detector_backend.fit(x_train, x_train, 
#                     **fit_params)

## plot autoencoder
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(1, n + 1):
#     # Display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i].reshape(48, 48))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # Display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i].reshape(48, 48))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()


tau_range = np.array([.03])*np.prod(input_shape[:2])

cue = loc_model(input_shape=input_shape,
                detector_backend=detector_backend,
                discriminator=discriminator,
                tau_range=tau_range)


cue.fit(X_train=x_train, y_train=y_train,
            optimizer=SGDW(lr=1., weight_decay=.0001, momentum=.9), fit_params=fit_params)

# cue.path_plot(plt2_params={'cmap': 'OrRd'})
# cue.DA_plot(X=x_train, y=y_train, threshold=1e-5)