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

np.random.seed(0)

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

timepoint = list(range(input_shape[0]))

import seaborn as sns
# sns.set_context("notebook", font_scale=.8, rc={"lines.linewidth": 1.})
# sns.set_style("whitegrid")
# custom_style = {
#             'grid.color': '0.8',
#             'grid.linewidth': 0.0,
# }
# sns.set_style(custom_style)
sns.set_theme(style= 'white', palette=None)

n_demo = 2
S_ind = np.array([np.random.choice(np.where(y[:,1] == 1)[0]) for i in range(n_demo)])
V_ind = np.array([np.random.choice(np.where(y[:,2] == 1)[0]) for i in range(n_demo)])
plot_df = {'time': [], 'ECG signal': [], 'label': [], 'sample_id': []}

for i in range(n_demo):
    plot_df['time'].extend(timepoint)
    plot_df['time'].extend(timepoint)
    plot_df['ECG signal'].extend(list(X[S_ind][i][:,0]))
    plot_df['ECG signal'].extend(list(X[V_ind][i][:,0]))
    plot_df['label'].extend(['S']*len(timepoint))
    plot_df['label'].extend(['V']*len(timepoint))
    plot_df['sample_id'].extend([str(i)]*len(timepoint))
    plot_df['sample_id'].extend([str(i)]*len(timepoint))

plot_df = pd.DataFrame(plot_df)

pal = dict(V="darkred", S="darkslateblue")

g = sns.relplot(
    data=plot_df,
    x="time", y="ECG signal", col='sample_id', row="label", hue='label', units="sample_id", 
    estimator=None, color=".6", kind="line", palette=pal, legend=True, row_order=['V', 'S']
)
g.despine(left=True)
sns.move_legend(g, "lower center", ncol=2, title=None, frameon=False)
plt.show()
