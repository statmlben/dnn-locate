fig = plt.figure(constrained_layout=False)
heights = [1]*rows
heights.append(.1)
spec = fig.add_gridspec(ncols=cols, nrows=rows+1, height_ratios=heights)
for row in range(rows):
	for col in range(cols):
		# compute X_diff_tmp
		X_diff_tmp = X_diff_R[col,row]
		X_diff_tmp[np.where(X_diff_tmp<=1e-4)] = np.nan
		ax = fig.add_subplot(spec[row, col])
		im1 = ax.imshow(X_test_R[col,row], vmin=0, vmax=1, alpha=0.7, cmap='binary')
		ax.axis('off')
		im2 = ax.imshow(X_diff_tmp, vmin=0, vmax=1, alpha=0.7, cmap='Wistia')
		ax.axis('off')

x_ax = fig.add_subplot(spec[-1, :])
x_ax = sns.heatmap(R_square.reshape(1,cols), cmap='binary', linewidths=.1, vmin=0, vmax=1, annot=True, cbar=False)
x_ax.axis('off')
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.001, wspace=0.001, right=0.8)
cbar_ax1 = fig.add_axes([0.9, 0.1, 0.02, 0.7])
cbar_ax2 = fig.add_axes([0.85, 0.1, 0.02, 0.7])
fig.colorbar(im1, cax=cbar_ax1)
fig.colorbar(im2, cax=cbar_ax2)
plt.show()

