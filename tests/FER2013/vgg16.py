from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

def VGG16Model():
    model = Sequential()
    model.add(Conv2D(input_shape=(48,48,1),filters=64,kernel_size=(3,3),
                    padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=2048,activation="relu", 
                    kernel_regularizer=regularizers.l2(0.001), 
                    bias_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(.5))
    model.add(Dense(units=2048,activation="relu", 
                    kernel_regularizer=regularizers.l2(0.001),
                    bias_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(.5))

    model.add(Dense(units=7, activation="softmax"))

    return model