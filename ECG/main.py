import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout,MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras import Model, layers,Sequential,regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow_addons.optimizers import AdamW, SGDW

discriminator=tf.keras.models.load_model('./best_model/model3.h5')
discriminator.summary()

mit_train_path="./dataset/mitbih_train.csv"
mit_test_path="./dataset/mitbih_test.csv"

def create_pd(train_path,test_path):
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    train.columns=[x for x in range(188)]
    test.columns=[x for x in range(188)]
    return pd.concat([train,test], axis=0, join='inner').sort_index()

mit= create_pd(mit_train_path,mit_test_path)

X = np.asarray(mit.iloc[:,:187].values)
y = mit.iloc[:,187].values
y = to_categorical(y)

X = X.reshape(-1, 187, 1)
input_shape = X.shape[1:]

## Model
from BBox_detect import loc_model
## define the backend detector before TRELU activation

detector_backend = tf.keras.Sequential(
    [
        layers.Input(shape=(input_shape[0], input_shape[1])),
        layers.Conv1D(
            filters=32, kernel_size=5, padding="same", strides=1, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=5, padding="same", strides=1, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=5, padding="same", strides=1, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=5, padding="same", strides=1, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=5, padding="same"),
    ]
)

es_detect1 = ReduceLROnPlateau(monitor="loss", factor=0.382, min_lr=1e-6,
                    verbose=1, patience=3, mode="min")
es_detect2 = EarlyStopping(monitor='loss', mode='min', min_delta=.0001, 
                        verbose=1, patience=20, restore_best_weights=True)

fit_params={'callbacks': [es_detect1, es_detect2],
            'epochs': 200, 'batch_size': 128}

# tau_range = np.array([.015])*np.prod(input_shape)
tau_range = np.array([.18])*np.prod(input_shape)

## define framework
cue = loc_model(input_shape=input_shape,
                detector_backend=detector_backend,
                discriminator=discriminator,
                target_r_square='auto',
                r_metric='acc',
                # r_metric='loss',
                tau_range=tau_range)

cue.fit(X_train=X, y_train=y,
            optimizer=SGDW(learning_rate=.1, weight_decay=.0001, momentum=.9), fit_params=fit_params)

r2_model = cue.R_square(X, y)

# cue.detector.save(f'detector_r2_%d'%(r2_model*100))

detector = cue.detector
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()

class_dict = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
n_demo = 3

for k in [0,1,2,3,4]:
    demo_ind = np.array([np.random.choice(np.where(y[:,k] == 1)[0]) for i in range(n_demo)])
    X_demo = X[demo_ind]
    X_demo_detect = detector.predict(X_demo)
    X_demo_hl =  - (X_demo_detect - X_demo) / (X_demo + 1e-5)

    sns.set_theme()
    timepoint = list(range(input_shape[0]))

    sns.set_theme(style= 'white', palette=None)
    for i in range(len(X_demo)):
        X_tmp, X_detect_tmp, X_hl_tmp = X_demo[i], X_demo_detect[i], X_demo_hl[i]
        plt.figure(figsize=(16, 8), dpi=80)
        plt.title('detect results for a random sample from class %s' %class_dict[k])
        plt.imshow(X_hl_tmp[np.newaxis,:], cmap="OrRd", aspect='auto', alpha=0.3, 
                                            extent = (0, 187, 0, 1))
        plt.colorbar()
        plt.plot(timepoint, X_tmp, linewidth=2.5, alpha=.7, color='r', 
            label='Extracted ECG Beat')
        plt.plot(timepoint, X_detect_tmp, linewidth=1.5, alpha=.7, color='b', linestyle='--', 
            label='Extracted ECG Beat (AFTER removing detected feats)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

# timepoint = list(range(187))
# pd_dict = { 'timepoint': [], 'signal': [], 'class': [], 'type': [] }

# X_demo = cue.X_demo
# X_demo_detect = cue.detector.predict(X_demo)
# X_demo_hl =  - (X_demo_detect - X_demo) / (X_demo + 1e-5)

# for i in range(len(X_demo)):
#     X_tmp, X_detect_tmp, X_hl_tmp = X_demo[i], X_demo_detect[i], X_demo_hl[i]
#     pd_dict['timepoint'].extend(timepoint)
#     pd_dict['signal'].extend(list(X_tmp.flatten()))
#     pd_dict['class'].extend([i]*len(timepoint))
#     pd_dict['type'].extend(['before']*len(timepoint))

#     pd_dict['timepoint'].extend(timepoint)
#     pd_dict['signal'].extend(list(X_detect_tmp.flatten()))
#     pd_dict['class'].extend([i]*len(timepoint))
#     pd_dict['type'].extend(['after']*len(timepoint))

#     pd_dict['timepoint'].extend(timepoint)
#     pd_dict['signal'].extend(list(X_hl_tmp.flatten()))
#     pd_dict['class'].extend([i]*len(timepoint))
#     pd_dict['type'].extend(['hl']*len(timepoint))

# df_plot = pd.DataFrame(pd_dict)

# sns.set_theme(style= 'white', palette=None)
# for i in range(len(X_demo)):
#     X_tmp, X_detect_tmp, X_hl_tmp = X_demo[i], X_demo_detect[i], X_demo_hl[i]
#     plt.figure()
#     plt.title('detect results for a random sample from class %s' %i)
#     plt.imshow(X_hl_tmp[np.newaxis,:], cmap="OrRd", aspect='auto', alpha=0.3, 
#                                         extent = (0, 187, 0, 1))
#     plt.colorbar()
#     plt.plot(timepoint, X_tmp, linewidth=2.5, alpha=.7, color='r', label='before')
#     plt.plot(timepoint, X_detect_tmp, linewidth=1.5, alpha=.7, color='b', linestyle='--', label='after')
#     plt.legend(loc='best')
#     plt.tight_layout()
#     plt.show()

