from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras import regularizers


def CAE(input_shape, c_hid=32):

    input_img = keras.Input(shape=input_shape)

    x = layers.Conv2D(c_hid, (3, 3), activation='relu', padding='same')(input_img) #32x32 -> 32x32
    x = layers.MaxPooling2D((2, 2), padding='same')(x) #32x32 -> 16x16
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(c_hid, (3, 3), activation='relu', padding='same')(x) #16x16 -> 16x16
    x = layers.MaxPooling2D((2, 2), padding='same')(x) #16x16 -> 8x8
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(2*c_hid, (3, 3), activation='relu', padding='same')(x) #8x8 -> 8x8
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x) #8x8 -> 4x4
    encoded = layers.BatchNormalization()(encoded)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    hid_feats = layers.Flatten()(encoded)
    # hid_feats = layers.Dense(2*np.prod(input_shape[:2]), activation='relu')(hid_feats)
    # hid_feats = layers.BatchNormalization()(hid_feats)
    hid_feats = layers.Dense(np.prod(input_shape[:2]), activation='relu')(hid_feats)
    hid_feats = layers.BatchNormalization()(hid_feats)
    decoded = layers.Reshape((input_shape[0], input_shape[1], 1))(hid_feats)
    # x = layers.Conv2D(2*c_hid, (3, 3), activation='relu', padding='same')(hid_map) 
    # x = layers.Conv2DTranspose(2*c_hid, (3, 3), activation='relu', padding='same')(hid_map)
    # x = layers.Conv2DTranspose(c_hid, (3, 3), strides=1, activation="relu", padding="same")(x)
    # decoded = layers.Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)

    # x = layers.Conv2D(2*c_hid, (3, 3), activation='relu', padding='same')(hid_map) 
    # x = layers.UpSampling2D((2, 2))(x) #4x4 -> 8x8
    # x = layers.Conv2D(c_hid, (3, 3), activation='relu', padding='same')(x)
    # x = layers.UpSampling2D((2, 2))(x) # 8x8 -> 16x16
    # x = layers.Conv2D(c_hid, (3, 3), activation='relu', padding='same')(x)
    # x = layers.UpSampling2D((2, 2))(x) 
    # decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    return autoencoder

def AE(input_shape, c_hid=32, latent_dim=100):

    input_img = keras.Input(shape=input_shape)
    # encoder
    x = layers.Conv2D(2*c_hid, (3, 3), activation="relu", padding="same")(input_img) # 48x48 -> 24x24
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(c_hid, (3, 3), activation="relu", padding="same")(x) ## 24x24 -> 12x12
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(c_hid/2, (3, 3), activation="relu", padding="same")(x) ## 12x12 -> 6x6
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # x = layers.Reshape((input_shape[0], input_shape[1], c_hid))(x)
    # Decoder
    x = layers.Dense(36*2*c_hid, activation='relu')(x)
    x = layers.Reshape((6, 6, 2*c_hid))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(2*c_hid, (3, 3), strides=4, activation="relu", padding="same")(x) # 6-> 24
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(c_hid, (3, 3), strides=2, activation="relu", padding="same")(x) # 24 -> 48
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(c_hid/2, (3, 3), strides=1, activation="relu", padding="same")(x) # 48 -> 48
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(1, (3, 3), strides=1, activation="sigmoid", padding="same",
                                kernel_regularizer=regularizers.l1(.001),
                                bias_regularizer=regularizers.l1(.001))(x)

    # Autoencoder
    autoencoder = keras.Model(input_img, x)
    return autoencoder