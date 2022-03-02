import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

def create_pd(train_path,test_path):
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    train.columns=[x for x in range(188)]
    test.columns=[x for x in range(188)]
    return pd.concat([train,test], axis=0, join='inner').sort_index()


mit_train_path="./dataset/mitbih_train.csv"
mit_test_path="./dataset/mitbih_test.csv"

mit= create_pd(mit_train_path,mit_test_path)
mit.head()

def create_k_folds_column(df):
    df.loc[:,'kfold']=-1
    df=df.sample(frac=1).reset_index(drop=True)
    y=df.loc[:,187].values
    kf=StratifiedKFold(n_splits=5)
    for fold,(target,index) in enumerate(kf.split(X=df,y=y)):
        df.loc[index,'kfold']=fold
    return df    

mit= create_k_folds_column(mit)
mit.head()

mit.loc[:,187].astype('int').value_counts()

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout,MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras import Model, layers,Sequential,regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

def make_model(X_train):
    model= Sequential()
    model.add(Convolution1D(32,5,activation='relu',input_shape=(187,1)))
    model.add(Convolution1D(64,5,activation='relu'))         
    model.add(MaxPooling1D(3))
    model.add(Convolution1D(128, 3, activation='relu'))
    model.add(Convolution1D(256, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    return model

def training_data(train,valid):
    X_train=np.asarray(train.iloc[:,:187].values)
    y_train=train.iloc[:,187].values
    X_valid=np.asarray(valid.iloc[:,:187].values)
    y_valid=valid.iloc[:,187].values
    X_train=tf.expand_dims(X_train, axis=2)
    X_valid=tf.expand_dims(X_valid, axis=2)
    y_train=to_categorical(y_train)
    y_valid=to_categorical(y_valid)
    return X_train,y_train,X_valid,y_valid

Epochs=100
Batch_size=64
my_callbacks = [EarlyStopping(patience=3,monitor='val_loss', mode='min',restore_best_weights=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=2, min_lr=0.00001, mode='auto')]
dict_acc={}
dict_acc2={}


def run_train(fold):
    train=mit[mit["kfold"]!=fold].reset_index(drop=True)
    valid=mit[mit["kfold"]==fold].reset_index(drop=True)
    X_train,y_train,X_valid,y_valid=training_data(train,valid)
    model=make_model(X_train)
    history = model.fit(X_train,y_train,validation_split=0.1,batch_size=Batch_size,epochs=Epochs,callbacks=my_callbacks)
    model.save(f'model{fold}.h5')
    results = model.evaluate(X_valid, y_valid)
    print("Test Accuracy: {:.2f}%".format(results[1] * 100))
    print("     Test AUC: {:.4f}".format(results[2]))
    dict_acc[f"{i}"]= "Test Accuracy: {:.2f}%".format(results[1] * 100) 
    

for i in range(5):
    print(f"{i}-fold trained",sep="/n")
    run_train(i)
    print("_______________________________",sep='/n')
    print("_______________________________",sep='/n')

print(dict_acc)
mit_model=tf.keras.models.load_model('model3.h5')
mit_model.summary()

y_pred=mit_model.predict(X_valid)

import itertools
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],normalize=True,
                      title='Confusion matrix, with normalization')
plt.show()
