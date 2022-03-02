## Ben Dai
wandb_flag = False

import wandb
if wandb_flag:
    wandb.init(project="ECG-DF-detection", entity="bdai")
    WANDB_NOTEBOOK_NAME='ECG-DF-detector: S and V'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import Model, layers,Sequential,regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow_addons.optimizers import AdamW, SGDW

np.random.seed(8)
# detector.summary()

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
n_label = y.shape[1]

import numpy as np
import pandas as pd
import seaborn as sns

class_dict = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
n_demo = 2
timepoint = list(range(input_shape[0]))

sns.set_theme(style= 'white', palette=None)

demo_ind = np.array([np.random.choice(np.where(y[:,k] == 1)[0]) for i in range(n_demo) for k in [1,2]])
print('show instances with indices: %s' %demo_ind)
X_demo = X[demo_ind]
y_demo = y[demo_ind]

# detetor_folders = ['detector_r2_10', 'detector_r2_50', 
#                     'detector_r2_60', 'detector_r2_70', 'detector_r2_75']

detetor_folders = ['detector_r2_70']

X_demo_detect, X_demo_hl = [], []
for detetor_folder in detetor_folders:
    detector=tf.keras.models.load_model(detetor_folder)
    X_demo_detect_tmp = detector.predict(X_demo)
    X_demo_detect.append(X_demo_detect_tmp)
    X_demo_hl_tmp = - (X_demo_detect_tmp - X_demo) / (X_demo + 1e-5)
    X_demo_hl.append(X_demo_hl_tmp)

X_demo_detect, X_demo_hl = np.array(X_demo_detect), np.array(X_demo_hl)

fig, axs = plt.subplots(2,2,sharex=True, sharey=True, gridspec_kw={'hspace': 0.1, 'wspace': 0.05})
for i in range(len(X_demo)):
    X_tmp, y_tmp = X_demo[i], np.argmax(y_demo[i])
    for j in range(len(detetor_folders)):
        r = int(i/n_demo)
        X_detect_tmp, X_hl_tmp = X_demo_detect[j,i], X_demo_hl[j,i]
        # plt.figure(figsize=(16, 8), dpi=80)
        if y_tmp == 0:
            cmap_tmp, color_tmp1, color_tmp2 = "Oranges", 'darkorange', 'darkgreen'
        elif y_tmp == 1:
            cmap_tmp, color_tmp1, color_tmp2 = "Purples", 'darkslateblue', 'darkred'
            axs[0,r].imshow(X_hl_tmp[np.newaxis,:], cmap=cmap_tmp, aspect='auto', alpha=0.7, 
                                            extent = (0, 187, 0, 1))
            axs[0,r].plot(timepoint, X_tmp, linewidth=2, alpha=.7, color=color_tmp1, label='S')
            axs[0,r].title.set_text('Extracted ECG: (sample %s, class %s)' %(r, class_dict[y_tmp]))
            axs[0,r].axis('off')
        elif y_tmp == 2:
            cmap_tmp, color_tmp1, color_tmp2 = "Reds", 'darkred', 'darkslateblue'
            axs[1,r].imshow(X_hl_tmp[np.newaxis,:], cmap=cmap_tmp, aspect='auto', alpha=0.7, 
                                            extent = (0, 187, 0, 1))
            axs[1,r].plot(timepoint, X_tmp, linewidth=2, alpha=.7, color=color_tmp1, label='V')
            axs[1,r].title.set_text('Extracted ECG: (sample %s, class %s)' %(r, class_dict[y_tmp]))
            axs[1,r].axis('off')
        elif y_tmp == 3:
            cmap_tmp, color_tmp1, color_tmp2 = "Blues", 'darkblue', 'darkred'
        elif y_tmp == 4:
            cmap_tmp, color_tmp1, color_tmp2 = "Greens", 'darkgreen', 'darkorange'
        # axs[i].imshow(X_hl_tmp[np.newaxis,:], cmap=cmap_tmp, aspect='auto', alpha=0.7, 
        #                                     extent = (0, 187, 0, 1))
        # axs[i].clim(0, 1)
        # axs[i].colorbar()
        # axs[i].plot(timepoint, X_tmp, linewidth=2.5, alpha=.7, color=color_tmp1, 
        #     label='Extracted ECG Beat')
        # plt.plot(timepoint, X_detect_tmp, linewidth=1.5, alpha=.7, color=color_tmp2, linestyle='--', 
        #     label='Extracted ECG Beat (AFTER removing detected signals with weights)')
        # plt.legend(loc='best')
        # axs[i].tight_layout()
        if wandb_flag:
            wandb.log({'ECG DF-detection': plt})
axs[0,r].legend(loc="upper right")
axs[1,r].legend(loc="upper right")
# for ax in axs.flat:
#     ax.set(ylabel='Extracted ECG signal')
# for ax in axs.flat:
#     ax.label_outer()
plt.tight_layout()
plt.show()
