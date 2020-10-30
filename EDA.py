import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
import numpy as np
import pandas as pd

def show_samples(R_square, X_test_R, X_test_noise_R, method='mask'):
	""" Plots generalized partial R values and its corresponding images.
	Parameters
	----------
	R_square: (numpy.array)
		List of arrays of generalized partial R values values. 
	X_test_R: (numpy.array)
		The original image we want to demostrate over different R_sqaure, shape is (#R_quare, #labels, **shape of image)
	X_test_noise_R: (numpy.array)
		The noised image we want to demostrate over different R_sqaure, shape is (#R_quare, #labels, **shape of image)
	"""
	cols, rows = X_test_R.shape[0], X_test_R.shape[1]
	if method == 'mask':
		X_diff_R = - (X_test_noise_R - X_test_R) / (X_test_R+1e-9)
	elif method == 'noise':
		X_diff_R = X_test_noise_R - X_test_R
	fig = plt.figure(constrained_layout=False)
	heights = [1]*rows
	heights.append(.06)
	spec = fig.add_gridspec(ncols=cols, nrows=rows+1, height_ratios=heights)
	for row in range(rows):
		for col in range(cols):
			# compute X_diff_tmp
			X_diff_tmp = X_diff_R[col,row]
			X_diff_tmp[np.where(np.abs(X_diff_tmp)<=1e-3)] = np.nan
			ax = fig.add_subplot(spec[row, col])
			im1 = ax.imshow(X_test_R[col,row], vmin=0, vmax=1, cmap='binary')
			ax.axis('off')
			im2 = ax.imshow(X_diff_tmp, vmin=0, vmax=1, cmap='OrRd')
			ax.axis('off')
	x_ax = fig.add_subplot(spec[-1, :])
	x_ax = sns.heatmap(R_square.reshape(1,cols), cmap='binary', linewidths=.00, vmin=0, vmax=1, annot=True, cbar=False)
	x_ax.axis('off')
	plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.0001, wspace=0.0001, right=0.82)
	cbar_ax1 = fig.add_axes([0.9, 0.1, 0.015, 0.7])
	cbar_ax2 = fig.add_axes([0.85, 0.1, 0.015, 0.7])
	fig.colorbar(im1, cax=cbar_ax1)
	fig.colorbar(im2, cax=cbar_ax2)
	# fig.text(0.5, 0.00, 'generalized partial R_sqaure', ha='center', va='center')
	plt.show()


def R_sqaure_path(lam_range, norm_lst, norm_test_lst, R_square_train_lst, R_square_test_lst=None):
	sns.set()
	R_train = pd.DataFrame({'tau': lam_range, 'l1-norm': norm_lst, 'R_square': R_square_train_lst, 'Type': ['R_square_train']*len(lam_range)})
	R_test = pd.DataFrame({'tau': lam_range, 'l1-norm': norm_test_lst, 'R_square': R_square_test_lst, 'Type': ['R_square_test']*len(lam_range)})
	df = pd.concat([R_train, R_test])
	sns.lineplot(data=df, x="tau", y="R_square", color='k', markers=True, alpha=.7, style='Type', lw=2.)
	plt.show()
	# sns.lineplot(data=df, x="l1-norm", y="R_square", color='k', markers=True, alpha=.7, style='Type', lw=2.)
	# plt.show()