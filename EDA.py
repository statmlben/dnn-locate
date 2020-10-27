import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
import numpy as np
import pandas as pd

def show_samples(X_test, X_test_noise, num_figs=3, method='mask'):
	ind = [randint(0,len(X_test)-1) for i in range(num_figs)]
	if method == 'mask':
		X_diff = np.nan_to_num( (X_test_noise - X_test) / X_test)
	elif method == 'noise':
		X_diff = X_test_noise[0,:,:,0] - X_test[0,:,:,0]
	for i in range(num_figs):
		fig = plt.figure(figsize = (10,5))
		ax_tmp = [fig.add_subplot(1,3,i+1) for i in range(3)]
		ax_tmp[0].imshow(X_test[ind[i],:,:,0], vmin=0, vmax=1)
		ax_tmp[0].axis('off')
		ax_tmp[1].imshow(X_test_noise[ind[i],:,:,0], vmin=0, vmax=1)
		ax_tmp[1].axis('off')
		ax_tmp[2].imshow(X_diff[ind[i],:,:,0], vmin=0, vmax=1)
		ax_tmp[2].axis('off')
	plt.show()

def R_sqaure_path(lam_range, norm_lst, norm_test_lst, R_square_train_lst, R_square_test_lst=None):
	sns.set()
	R_train = pd.DataFrame({'lam': lam_range, 'l1-norm': norm_lst, 'R_square': R_square_train_lst, 'Type': ['R_square_train']*len(lam_range)})
	R_test = pd.DataFrame({'lam': lam_range, 'l1-norm': norm_test_lst, 'R_square': R_square_test_lst, 'Type': ['R_square_test']*len(lam_range)})
	df = pd.concat([R_train, R_test])
	sns.lineplot(data=df, x="lam", y="R_square", color='k', markers=True, alpha=.7, style='Type', lw=2.)
	plt.show()
	sns.lineplot(data=df, x="l1-norm", y="R_square", color='k', markers=True, alpha=.7, style='Type', lw=2.)
	plt.show()