"""
Discriminative feature localization for deep learning models
"""

# Author: Ben Dai <bdai@umn.edu>

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Reshape, Flatten, Add, Multiply
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import numpy as np
from random import randint
import seaborn as sns
import pandas as pd

class loc_model(object):
    """
    class for discriminative feature detection for deep learning models.
    
    Parameters
    ----------

    input_shape : {tuple-like} (shape of the feature/image)
     For example, in MNIST dataset ``input_shape = (28, 28, 1)``.
    
    localizer_backend: {keras-defined neural network}
     A backend neural network before reshape and Truncated ReLU activation.

    discriminator: {keras-defined neural network}
     A pretrained neural network needs to be explained.

    tau_range : {list-like}
     List of tau to define the localizers.
    
    target_r_square: {float, [0,1]}
     A target R_square to be explained. 

    R_square: {list-like}
     Records for R_square values based on a dataset.

    """
    def __init__(self, input_shape, discriminator, localizer_backend=None, tau_range = np.arange(1,10), target_r_square='auto', r_metric='acc', activation='tanh+relu', save_path=False):
        # self.labels = labels
        self.input_shape = input_shape
        self.tau_range = tau_range
        self.discriminator = discriminator
        self.target_r_square = target_r_square
        self.r_square = []
        self.localizer_backend = localizer_backend
        self.localizer = None
        self.combined = None
        self.X_demo = None
        self.loss = tf.keras.losses.get(self.discriminator.loss)
        self.X_diff_demo = []
        self.r_metric=r_metric
        self.activation = activation
        self.save_path = save_path

        # labels: integer, (number of labels for classification, dimension of outcome for regression)
        #  For example, in MNIST dataset ``labels = 10``.

    def build_localizer(self, tau=10., max_value=1.):
        """
        Building a localizer for the proposed framework

        Parameters
        ----------

        tau: {float}
         The magnitude of the localizer.

        max_value: {float, (0, 1]}
         The maximum proportion of the localizer.
        """
        self.localizer_backend.layers[-1].activation = tf.keras.layers.Activation('linear')
        X = Input(shape=self.input_shape)
        noise = self.localizer_backend(X)
        noise = tf.keras.layers.BatchNormalization()(noise)
        noise = tf.keras.layers.Softmax(axis=(1,2))(noise)
        noise = tau*noise
        if self.activation == 'TReLU':
            noise = backend.relu(noise, max_value=max_value)
        elif self.activation == 'tanh+relu':
            noise = backend.tanh(backend.relu(noise))
        elif self.activation == 'ReLU':
            noise = backend.relu(noise)
        else:
            raise Exception("Sorry, activation currently must be ReLU, TReLU or tanh+relu.") 
        # noise = backend.tanh(noise)
        X_mask = Multiply()([noise, X])
        X_noise = Add()([X, -X_mask])
        self.localizer = Model(X, X_noise)


    def build_combined(self, optimizer=Adam(learning_rate=.0005)):
        """
        Building a localizer and a combined model for the proposed framework

        Parameters
        ----------
        
        localizer: {keras-defined neural network}
         A neural network for localizer.

        optimizer: {keras-defined optimizer: ``tensorflow.keras.optimizers``}, default = 'SGD(lr=.0005)'
         A optimizer used to train the localizer.
         
        """

        def neg_loss(y_true, y_pred):
            return -self.loss(y_true, y_pred)

        # The localizer generate X_noise based on input imgs
        X = Input(shape=self.input_shape)
        X_noise = self.localizer(X)

        # For the combined model we will only train the localizer
        self.discriminator.trainable = False

        # The discriminator takes noised images as input and determines probs
        prob = self.discriminator(X_noise)

        # The combined model  (stacked localizer and discriminator)
        # Trains the localizer to attack the discriminator
        self.combined = Model(X, prob)

        self.combined.compile(loss=neg_loss, optimizer=optimizer)

    def fit(self, X_train, y_train, fit_params, datagen=None, demo_ind=None, X_test=None, y_test=None, optimizer=SGD(lr=.0005)):
        """
        Fitting the localizer based on a given dataset.

        Parameters
        ----------

        X_train : {array-like} of shape (n_samples, dim_features)
         Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
        
        y_train : {array-like} of shape (n_samples,)
         Output vector/matrix relative to X.
        
        X_test : {array-like} of shape (n_samples, dim_features), default = None
         Instances features to compute the r_square. If ``None``, X_test = X_train
        
        y_test : {array-like} of shape (n_samples,), default = None
         Output to compute the r_square. If ``None``, y_test = y_train
        
        localizer : {keras-defined neural network}
         A neural network for localizer before Truncated RELU activation.
        
        fit_params: {dict of fitting parameters}
         See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``,
          ``validation_split``, ``validation_data``, and so on.
        
        demo_ind : {array-like} default=None
         The index set for demonstrated instances.
        
        optimizer: {keras-defined optimizer: ``tensorflow.keras.optimizers``}, default = 'SGD(lr=.0005)'
         A optimizer used to train the localizer.
        """

        n_label = y_train.shape[1]
        ## load X_demo
        if (X_test is None) or (y_test is None):
            if demo_ind is None:
                demo_ind = np.array([np.where(y_train[:,k] == 1)[0][0] for k in range(n_label)])
                # demo_ind = np.array(demo_ind)
                self.X_demo = X_train[demo_ind]
            else:
                demo_ind = np.array(demo_ind)
                self.X_demo = X_train[demo_ind]
        else:
            if demo_ind is None:
                demo_ind = np.array([np.where(y_test[:,k] == 1)[0][0] for k in range(n_label)])
                # demo_ind = [np.where(y_test==k)[0][0] for k in list(set(y_test))]
                # demo_ind = np.array(demo_ind)
                self.X_demo = X_test[demo_ind]
            else:
                demo_ind = np.array(demo_ind)
                self.X_demo = X_test[demo_ind]
        ## determine target_r_square
        if self.target_r_square == 'auto':
            loss_base, acc_base = self.discriminator.evaluate(X_train, y_train)[:2]
            loss_random, acc_random = self.discriminator.evaluate(X_train, np.random.permutation(y_train))[:2]
            if self.r_metric=='acc':
                self.target_r_square = 1. - (1. - acc_base) / (1. - acc_random)
            else:
                self.target_r_square = 1. - loss_base / loss_random

        ## fitting the models
        reach_target = 0
        for tau_tmp in self.tau_range:
            self.build_localizer(tau=tau_tmp)
            self.build_combined(optimizer=optimizer)
            if datagen == None:
                self.combined.fit(X_train, y_train, **fit_params)
            else:
                self.combined.fit(datagen.flow(X_train, y_train, batch_size = fit_params['batch_size']), **fit_params)
            if (X_test is None) or (y_test is None):
                r_square_tmp = self.R_square(X_train, y_train)
            else:
                r_square_tmp = self.R_square(X_test, y_test)
            # store the X_demo for the localizer
            X_diff_demo_tmp = self.locate(self.X_demo)
            self.r_square.append(r_square_tmp)
            self.X_diff_demo.append(X_diff_demo_tmp)
            if self.save_path:
                self.localizer.save(f'./saved/loc_r2_%d'%(r2_model*100))

            if r_square_tmp >= self.target_r_square:
                reach_target = 1
                print('early stop in tau = %.3f, R2: %.3f; target R2: %.3f is reached' %(tau_tmp, r_square_tmp, self.target_r_square))
                break
        if reach_target == 0:
            print('the localizer can not reach the target r_square, pls increase tau in tau_range.')
        self.X_diff_demo = np.array(self.X_diff_demo)
        self.r_square = np.array(self.r_square)

    def locate(self, X):
        """
        Return the localized discriminative features by the fitted localizer.

        Parameters
        ----------

        X : {array-like} of shape (n_samples, dim_features)
            Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.

        Return
        ------
        X_diff : {array-like} of shape (n_samples, dim_features)
            Attacking features generated by the localizer
        """

        X_noise = self.localizer.predict(X)
        X_diff = - (X_noise - X) / (X + 1e-7)
        # X_diff = X_diff.mean(axis=-1).reshape(-1, self.input_shape[0], self.input_shape[1], 1)
        X_diff = X_diff.mean(axis=-1)
        return X_diff

    def R_square(self, X, y):
        """
        Report R_square for the fitted localizer based on a given dataset
        
        Parameters
        ----------

        X : {array-like} of shape (n_samples, dim_features)
         Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.

        y : {array-like} of shape (n_samples,)
         Output vector/matrix relative to X.

        Return
        ------
        
        R_square: array of float [0, 1]
            The R_square for fitted localizer based on a given dataset.
        """
        print('#'*50)
        print('compute the R2 for the fitted localizer.')
        print('#'*50)

        ## only 
        loss_base, acc_base = self.discriminator.evaluate(X, y)[:2]
        X_noise = self.localizer.predict(X)
        loss_noise, acc_noise = self.discriminator.evaluate(X_noise, y)[:2]
        if self.r_metric == 'acc':
            R_square = 1. - (1. - acc_base) / (1. - acc_noise)
        else:
            R_square = 1. - loss_base / loss_noise
        return R_square

    def R_sqaure_path(self):
        """
        Plot solution path for the proposed method wrt tau_range
        """
        sns.set()
        R_dict = pd.DataFrame({'tau': self.tau_range[:len(self.r_square)], 'R_square': self.r_square})
        sns.lineplot(data=R_dict, x="tau", y="R_square", color='k', markers=True, alpha=.7, lw=2.)
        plt.show()

    def path_plot(self, threshold=None, plt1_params={'cmap': 'binary'}, plt2_params={'cmap': 'OrRd'}):
        """
        Plots generalized partial R values and its corresponding images wrt tau_range.
        
        Parameters
        ----------
        threshold : {array-like} or float
         threshold to truncate the small detected pixels
        
        plt1_params : {dict-like} 
         dict for imshow for X_demo

        plt2_params : {dict-like}
         dict for imshow for X_diff_demo

        """
        cols, rows = self.X_diff_demo.shape[0], self.X_diff_demo.shape[1]
        fig = plt.figure(constrained_layout=False)
        heights = [1]*rows
        heights.append(.06)
        spec = fig.add_gridspec(ncols=cols, nrows=rows+1, height_ratios=heights)
        for row in range(rows):
            for col in range(cols):
                if threshold is None:
                    threshold_tmp = 1e-3
                elif isinstance(threshold, float):
                    threshold_tmp = threshold
                else:
                    threshold_tmp = threshold[col]
                # compute X_diff_tmp
                X_diff_tmp = self.X_diff_demo[col,row]
                X_diff_tmp[np.where(np.abs(X_diff_tmp)<=threshold_tmp)] = 0.
                ax = fig.add_subplot(spec[row, col])
                im1 = ax.imshow(self.X_demo[row], vmin=0, vmax=1, **plt1_params)
                ax.axis('off')
                im2 = ax.imshow(X_diff_tmp, vmin=0, vmax=1, alpha=X_diff_tmp.mean(axis=-1), **plt2_params)
                ax.axis('off')
        x_ax = fig.add_subplot(spec[-1, :])
        x_ax = sns.heatmap(self.r_square.reshape(1,cols), 
                            cmap='binary', linewidths=.00, vmin=0, vmax=1, 
                            annot=True, cbar=False)
        x_ax.axis('off')
        plt.subplots_adjust(top = 0.99, bottom=0.01, 
                            hspace=0.0001, wspace=0.0001, right=0.82)
        cbar_ax1 = fig.add_axes([0.9, 0.1, 0.015, 0.7])
        cbar_ax2 = fig.add_axes([0.85, 0.1, 0.015, 0.7])
        fig.colorbar(im1, cax=cbar_ax1)
        fig.colorbar(im2, cax=cbar_ax2)
        # fig.text(0.5, 0.00, 'generalized partial R_sqaure', ha='center', va='center')
        plt.show()


    def DA_plot(self, X, y, demo_ind=None, threshold=None, plt1_params={'cmap': 'binary'}, plt2_params={'cmap': 'OrRd'}):
        """ 
        Plots data-adaptive detected region for the fitted localizer.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, dim_features)
         Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.

        y : {array-like} of shape (n_samples,)
         Output vector/matrix relative to X.

        demo_ind : {array-like} of shape (num_instance, num_labels) default=None

        threshold : {array-like} or float, default=None
         threshold to truncate the small detected pixels        

        """
        X_demo, X_diff = [], []
        n_sample, n_label = y.shape

        if demo_ind is None:
            for i in range(5):
                demo_ind_tmp = np.array([np.where(y[:,k] == 1)[0][i] for k in range(n_label)])
                X_demo.append(X[demo_ind_tmp])
                X_diff.append(self.locate(X[demo_ind_tmp]))
        else:
            for i in range(len(demo_ind)):
                demo_ind_tmp = demo_ind[i]
                X_demo.append(X[demo_ind_tmp])
                X_diff.append(self.locate(X[demo_ind_tmp]))

        X_demo = np.array(X_demo)
        X_diff = np.array(X_diff)
        cols, rows = X_demo.shape[0], X_demo.shape[1]

        fig = plt.figure(constrained_layout=False)
        spec = fig.add_gridspec(ncols=cols, nrows=rows)
        for row in range(rows):
            for col in range(cols):
                if threshold is None:
                    threshold_tmp = 1e-3
                elif isinstance(threshold, float):
                    threshold_tmp = threshold
                else:
                    threshold_tmp = threshold[col]
                # compute X_diff_tmp
                X_diff_tmp = X_diff[col,row]
                X_diff_tmp[np.where(np.abs(X_diff_tmp)<=threshold_tmp)] = 0.
                ax = fig.add_subplot(spec[row, col])
                im1 = ax.imshow(X_demo[col,row], vmin=0, vmax=1, **plt1_params, aspect='auto')
                ax.axis('off')
                im2 = ax.imshow(X_diff_tmp, vmin=0, vmax=1, alpha=X_diff_tmp.mean(axis=-1), **plt2_params, aspect='auto')
                ax.axis('off')
        plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.1, wspace=0.1, right=0.82)
        cbar_ax1 = fig.add_axes([0.9, 0.1, 0.015, 0.7])
        cbar_ax2 = fig.add_axes([0.85, 0.1, 0.015, 0.7])
        fig.colorbar(im1, cax=cbar_ax1)
        fig.colorbar(im2, cax=cbar_ax2)
        # fig.text(0.5, 0.00, 'generalized partial R_sqaure', ha='center', va='center')
        plt.show()

