import numpy as np 
from sklearn import preprocessing, datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import os
from sklearn.decomposition import PCA , FastICA

data =datasets.load_breast_cancer()
x, y = data.data, data.target 

x = preprocessing.StandardScaler().fit_transform(x)
breakpoint()




def find_eigen_values( x, y):
	y = np.transpose(x)
	cov_mat = np.cov( [ y[k,:] for k in range( len(y))])
	eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
	for i in range(len(eig_val_cov)):
		eigvec_cov = eig_vec_cov[:,i].reshape(1,30).T
		#print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
	eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
	eig_pairs.sort(key=lambda x: x[0], reverse=True)
	eig_values= [lis[0] for lis in eig_pairs]
	plt.figure() 
	plt.bar(np.arange( 1, len(eig_values)+1),eig_values)
	plt.xlabel('Number of Components')
	plt.ylabel('Eigenvalues')
	plt.title(" Eigen values in Decreasing order")
	plt.grid()
	plt.show() 




def pca_analysis( x):
	scaler=preprocessing.StandardScaler()#instantiate
	scaler.fit(x) 
	X_scaled=scaler.transform(x)
	#print ("after scaling minimum", X_scaled.min(axis=0) )
	pca=PCA() 
	pca.fit(X_scaled) 
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('number of components')
	plt.ylabel('cumulative explained variance')
	plt.grid()
	plt.show()
	X_pca=pca.transform(X_scaled) 
	return X_pca



find_eigen_values( x, y)











