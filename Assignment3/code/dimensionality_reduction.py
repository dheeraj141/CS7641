import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import os
from sklearn.decomposition import PCA , FastICA
from numpy import linalg as LA
import scipy
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, silhouette_samples,mean_squared_error
import sys
import time
import matplotlib.cm as cm
import pandas as pd
from sklearn.metrics import homogeneity_score, completeness_score, \
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn import mixture
import itertools
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
from clustering import k_means, Kmeans_clustering,Kmeans_silhouette_analysis, gmm_analysis, gmm_using_silhouette, EM,k_means1
from sklearn.metrics import accuracy_score
from numpy import linalg as LA
from scipy.linalg import pinv
from classifier import neural_network
from sklearn.feature_selection import SelectKBest, chi2


data =datasets.load_breast_cancer()
x, y = data.data, data.target 


 


def prepare_dataSet( input_file):
    data_set= []
    f = open(input_file)
    for line in f:
        data_set.append(line.split(' '))
    
    resulting_data=[]
    for data in data_set:
        data1 = [x.strip(' ') for x in data]
        data2=[]
        for x in data1:
            if(len(x) >0):
                data2.append(float(x))
        
        resulting_data.append( data2)
    
    
    #return resulting_data
    return np.asarray(resulting_data)


def extract_labels( file):
    labels = []
    new_labels=[]
    f = open(file)
    for line in f:
        labels.append( line.rstrip())
    for x in labels:
        new_labels.append(int(x))
    #print(len(new_labels))
    new_labels = np.asarray(new_labels)
    return new_labels
    #print( data_set.shape, new_labels.shape)
    #new_labels = np.transpose(new_labels)
    #print( type(new_labels), type(data_set))
    



X_train1 = prepare_dataSet( '../DataSet/train/X_train.txt')
data2 = X_train1
X_train1 = preprocessing.StandardScaler().fit_transform(X_train1)
#X_train1 = preprocessing.scale(X_train1)
y_train1 = extract_labels('../DataSet/train/y_train.txt')
X_test1 = prepare_dataSet('../DataSet/test/X_test.txt')
X_test1= preprocessing.scale( X_test1)

y_test1 = extract_labels('../DataSet/test/y_test.txt')




def kmeans_accuracy( X,y):
	scaler=preprocessing.StandardScaler()#instantiate
	scaler.fit(x) 
	accuracy = []
	clusters = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
	X_scaled=scaler.transform(x)
	for i in range( 1, 15):
		k_means = KMeans(n_clusters = i, random_state=14, n_init=30)
		k_means.fit(X)
		c_labels = k_means.labels_
		accuracy.append( accuracy_score( c_labels, y))
	plt.plot(clusters, accuracy)
	plt.xlabel('Clusters')
	plt.ylabel('accuracy')
	plt.grid()
	plt.show()



def EM_accuracy( X,y):
	scaler=preprocessing.StandardScaler()#instantiate
	scaler.fit(x) 
	accuracy = []
	clusters = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
	X_scaled=scaler.transform(x)
	for i in range( 1, 15):
		gmm=mixture.GaussianMixture(i, n_init=20).fit(X) 
		labels=gmm.predict(X)
		accuracy.append( accuracy_score( labels, y))
	plt.plot(clusters, accuracy)
	plt.xlabel('Clusters')
	plt.ylabel('accuracy')
	plt.grid()
	plt.show()




def find_eigen_values( x, y):
	y = np.transpose(x)
	cov_mat = np.cov( [ y[k,:] for k in range( len(y))])
	eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
	for i in range(len(eig_val_cov)):
		eigvec_cov = eig_vec_cov[:,i].reshape(1,len(y)).T
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




def reconstruction_error_pca( x):
	scaler=preprocessing.StandardScaler()#instantiate
	scaler.fit(x) 
	X_scaled=scaler.transform(x)
	
	max_comp=150
	start=1
	error_record=[]
	for i in range(start,max_comp):
		pca = PCA(n_components=i, random_state=42)
		pca2_results = pca.fit_transform(x)
		pca2_proj_back=pca.inverse_transform(pca2_results)
		total_loss=LA.norm((x-pca2_proj_back),None)
		error_record.append(total_loss)
	plt.clf()
	plt.figure()
	plt.title("reconstruct error of pca")
	plt.plot(error_record,'r')
	plt.xticks(range(len(error_record)), range(start,max_comp), rotation='vertical')
	plt.xlim([-1, len(error_record)])
	plt.show()



def reconstruction_error_ica( x,n):
	scaler=preprocessing.StandardScaler()#instantiate
	scaler.fit(x) 
	X_scaled=scaler.transform(x)
	
	max_comp=n
	start=1
	error_record=[]
	for i in range(start,max_comp):
		pca = FastICA(n_components=i, random_state=42,max_iter =400, tol =0.01)
		pca2_results = pca.fit_transform(x)
		pca2_proj_back=pca.inverse_transform(pca2_results)
		total_loss=LA.norm((x-pca2_proj_back),None)
		error_record.append(total_loss)
	plt.clf()
	plt.figure(figsize=(5,5))
	plt.title("reconstruct error of Ica")
	plt.plot(error_record,'r')
	plt.xticks(range(len(error_record)), range(start,max_comp), rotation='vertical')
	plt.xlim([-1, len(error_record)])
	plt.grid()
	plt.show()


def reconstruction_error_rp( x):
	scaler=preprocessing.StandardScaler()#instantiate
	scaler.fit(x) 
	X_scaled=scaler.transform(x)
	
	max_comp=50
	start=1
	error_record=[]
	for i in range(start,max_comp):
		pca = FastICA(n_components=i, random_state=42)
		pca2_results = pca.fit_transform(x)
		pca2_proj_back=pca.inverse_transform(pca2_results)
		total_loss=LA.norm((x-pca2_proj_back),None)
		error_record.append(total_loss)
	plt.clf()
	plt.figure(figsize=(5,5))
	plt.title("reconstruct error of Ica")
	plt.plot(error_record,'r')
	plt.xticks(range(len(error_record)), range(start,max_comp), rotation='vertical')
	plt.xlim([-1, len(error_record)])
	plt.grid()
	plt.show()


def kurtosis_analysis(X, n):
	arr = []
	for i in range(1,n):
		dim_red = FastICA(n_components = i, random_state = 42, max_iter = 500, tol = 0.05).fit_transform(X)
		kurt = scipy.stats.kurtosis(dim_red)
		arr.append(np.mean(kurt))
	breakpoint()
	plt.plot(np.arange(1,n),arr)
	plt.xlabel('Number of Components')
	plt.ylabel('Kurtosis Value')
	plt.grid()
	plt.show()



def kurtosis_analysis_new(x, y):
	#x = preprocessing.StandardScaler().fit_transform(x)
	arr = []
	max_features = 10
	ica = FastICA(n_components = max_features)
	S = ica.fit_transform(x)

	#breakpoint()

	kurtosis_values = [ (scipy.stats.kurtosis(  S[:,k])) for k in range( max_features) ]
	kurtosis_pairs = [(np.abs(kurtosis_values[i]), i) for i in range(len(kurtosis_values))]
	kurtosis_pairs .sort(key=lambda x: x[0], reverse=True)
	x_values = [lis[1] for lis in kurtosis_pairs]
	y_values= [lis[0] for lis in kurtosis_pairs] 

	plt.figure() 

	plt.bar(np.arange( 1, max_features+1),y_values)
	plt.xlabel('Number of Components')
	plt.ylabel('Kurtosis Value')
	plt.grid()
	plt.show()
	accuracy = []

	for i in range( len(x_values)):
		data = S[:,x_values[:i+1]]
		#clustering_analysis_after_dimension( data, y)
		#k_means = KMeans(n_clusters = 2, random_state=14, n_init=30)
		#k_means.fit(data)
		#c_labels = k_means.labels_
		#accuracy.append( accuracy_score( c_labels, y))

	plt.figure() 

	plt.plot(np.arange( 1, max_features+1),accuracy)
	plt.xlabel('Number of Components')
	plt.ylabel('Kurtosis Value')
	plt.grid()
	plt.show()



def kurtosis_analysis_new1(x, y,n):
	#x = preprocessing.StandardScaler().fit_transform(x)
	arr = []
	max_features = n
	ica = FastICA(n_components=max_features, max_iter =500)
	S = ica.fit_transform(x)

	#breakpoint()

	kurtosis_values = [ (scipy.stats.kurtosis(  S[:,k])) for k in range( max_features) ]
	kurtosis_pairs = [(np.abs(kurtosis_values[i]), i) for i in range(len(kurtosis_values))]
	kurtosis_pairs .sort(key=lambda x: x[0], reverse=True)
	x_values = [lis[1] for lis in kurtosis_pairs]
	y_values= [lis[0] for lis in kurtosis_pairs] 

	plt.figure() 

	plt.bar(np.arange( 1, max_features+1),y_values)
	plt.xlabel('Number of Components')
	plt.ylabel('Kurtosis Value')
	plt.grid()
	plt.show()
	k_means( 2, S, y)
	EM( 2, S, y, 'spherical')
	EM( 2, S, y, 'tied')
	EM( 2, S, y, 'diag')
	EM( 2, S, y, 'full')
	#clustering_analysis_after_dimension( S, y)
	
	accuracy = []

	for i in range( len(x_values)):
		data = S[:,x_values[:i+1]]
		EM( )
		#clustering_analysis_after_dimension( data, y)
	



def RP_analysis(X):
	scaler=preprocessing.StandardScaler()#instantiate
	scaler.fit(X) 
	X=scaler.transform(X)
	arr5 = []
	for i in range(1,X.shape[1]):
		rp = GaussianRandomProjection(n_components=i, random_state=5)
		X_rp = rp.fit(X)
		p = pinv(X_rp.components_)
		w = X_rp.components_
		reconstructed = ((p@w)@(X.T)).T 
		arr5.append(mean_squared_error(X,reconstructed))

	arr5 = np.array(arr5)
	arr10 = []
	for i in range(1,X.shape[1]):
		rp = GaussianRandomProjection(n_components=i, random_state=10)
		X_rp = rp.fit(X)
		p = pinv(X_rp.components_)
		w = X_rp.components_
		reconstructed = ((p@w)@(X.T)).T 
		arr10.append(mean_squared_error(X,reconstructed))

	arr10 = np.array(arr10)
	arr15 = []
	for i in range(1,X.shape[1]):
		rp = GaussianRandomProjection(n_components=i, random_state=15)
		X_rp = rp.fit(X)
		p = pinv(X_rp.components_)
		w = X_rp.components_
		reconstructed = ((p@w)@(X.T)).T 
		arr15.append(mean_squared_error(X,reconstructed))

	arr15 = np.array(arr15)
	arr25 = []
	for i in range(1,X.shape[1]):
		rp = GaussianRandomProjection(n_components=i, random_state=25)
		X_rp = rp.fit(X)
		p = pinv(X_rp.components_)
		w = X_rp.components_
		reconstructed = ((p@w)@(X.T)).T 
		arr25.append(mean_squared_error(X,reconstructed))

	arr25 = np.array(arr25)
	plt.plot(np.arange(1, X.shape[1]),arr5, label = "5")
	plt.plot(np.arange(1, X.shape[1]),arr10,label = "10")
	plt.plot(np.arange(1, X.shape[1]),arr15,label = "15")
	plt.plot(np.arange(1, X.shape[1]),arr25,label = "25")
	plt.legend()
	plt.title('reconstruction error  vs. Different random seed')
	plt.xlabel('Number of Components')
	plt.ylabel('Reconstruction Error')
	plt.grid()
	plt.show()



def plotting_data2( X_pca, y):
	breakpoint()
	Xax=X_pca[:,0]
	Yax=X_pca[:,1]
	Zax = X_pca[:,2]
	labels=y
	cdict={1:'red',2:'green', 3:'blue', 4:'yellow',5:'magenta',6:'black'}
	labl={1:'walking',2:'Climbing up', 3: 'climbing down', 4: 'sitting',5:'standing', 6:'laying'}
	marker={1:'*',2:'o',3:'.',4:'v',5:'^',6:'<'}
	#alpha={0:.3, 1:.5}
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	fig.patch.set_facecolor('white')
	for l in np.unique(labels):
		ix=np.where(labels==l)
		ax.scatter(Xax[ix],Yax[ix],Zax[ix],c=cdict[l],s=40,label=labl[l],marker=marker[l])
	plt.xlabel("First Principal Component",fontsize=14)
	plt.ylabel("Second Principal Component",fontsize=14)
	ax.set_zlabel('Third principal component')
	plt.legend()
	breakpoint()
	plt.show()
	plt.matshow(pca.components_,cmap='viridis')
	plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
	plt.colorbar()
	plt.xticks(range(len(cancer.feature_names)),cancer.feature_names,rotation=65,ha='left')
	plt.tight_layout()
	plt.show()


def Plot_2d(Z,Y): 
    for i in range(len(Y)):
        if Y[i] == 0:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'r')
        elif Y[i] == 1:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'b')
        elif Y[i] == 2:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'g')
        elif Y[i] == 3:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'k')
        elif Y[i] == 4:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'c')
        elif Y[i] == 5:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'm')
        elif Y[i] == 6:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'y')
    plt.xlabel("First  Component",fontsize=10)
    plt.ylabel("Second  Component",fontsize=10)
    plt.show()




def clustering_analysis_after_dimension( x, y):
	#Kmeans_clustering(x )
	#Kmeans_silhouette_analysis( x, y)
	#kmeans_accuracy( x, y)
	#k_means1(2, x, y)
	#gmm_cluster = gmm_analysis( x,y)
	gmm_using_silhouette(x, y)
	#EM_accuracy( x, y)

def Plot_3d(Z,Y):
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(Y)):
        if Y[i] == 0:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = 'b', marker='o')
        elif Y[i] == 1:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = 'r', marker='o')
        elif Y[i] == 2:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = 'g', marker='o')
        elif Y[i] == 3:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = 'k', marker='o')
        elif Y[i] == 4:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = 'c', marker='o')
        elif Y[i] == 5:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = 'm', marker='o')
        elif Y[i] == 6:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = 'y', marker='o')
    
    plt.xlabel("First  Component",fontsize=10)
    plt.ylabel("Second  Component",fontsize=10)
    #ax.set_zlabel('Third principal component')
    plt.show()


def plotting_data1(X_pca, y):
	cancer = datasets.load_breast_cancer()
	Xax=X_pca[:,0]
	Yax=X_pca[:,1]
	labels=y
	cdict={0:'red',1:'green'}
	labl={0:'Malignant',1:'Benign'}
	marker={0:'*',1:'o'}
	alpha={0:.3, 1:.5}
	fig,ax=plt.subplots(figsize=(7,5))
	fig.patch.set_facecolor('white')
	for l in np.unique(labels):
		ix=np.where(labels==l)
		ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,label=labl[l],marker=marker[l],alpha=alpha[l])
	plt.xlabel("First Principal Component",fontsize=14)
	plt.ylabel("Second Principal Component",fontsize=14)
	plt.legend()
	plt.show()





def reconstruction_error_K_best( x,y,n):
	x = preprocessing.MinMaxScaler().fit_transform(x)
	
	max_comp=n
	start=1
	error_record=[]
	for i in range(start,max_comp):
		pca = SelectKBest(chi2, k=i)
		pca2_results = pca.fit_transform(x, y)
		pca2_proj_back=pca.inverse_transform(pca2_results)
		total_loss=LA.norm((x-pca2_proj_back),None)
		error_record.append(total_loss)
	plt.clf()
	plt.figure(figsize=(5,5))
	plt.title("reconstruct error of Ica")
	plt.plot(error_record,'r')
	plt.xticks(range(len(error_record)), range(start,max_comp), rotation='vertical')
	plt.xlim([-1, len(error_record)])
	plt.grid()
	plt.show()










def neural_network_analysis( x,y):

	x1= x
	scaler=preprocessing.StandardScaler()#instantiate
	scaler.fit(x) 
	x=scaler.transform(x)
	pca_acc = []
	pca_train = []
	pca_test = []
	pca_iter = []

	ica_acc = []
	ica_train = []
	ica_test = []
	ica_iter = []

	rp_acc = []
	rp_train = []
	rp_test = []
	rp_iter = []

	k_best_acc = []
	k_best_train = []
	k_best_test = []
	k_best_iter = []
	
	orig_acc = []
	orig_train = []
	orig_test = []
	orig_iter = []



	for i in range( 1, x.shape[1]):
		data_pca = PCA(n_components = i).fit_transform(x)
		X_train, X_test, y_train, y_test = train_test_split(data_pca, y, test_size=0.2, random_state=42)
		train, test, acc, iterations =neural_network( X_train, y_train, X_test, y_test, plotting = False)
		pca_acc.append( acc)
		pca_train.append( train)
		pca_test.append(test)
		pca_iter.append( iterations)

		data_ica = FastICA(n_components = i, max_iter = 400, tol = 0.0005).fit_transform(x)
		X_train, X_test, y_train, y_test = train_test_split(data_ica, y, test_size=0.2, random_state=42)
		train, test, acc, iterations= neural_network( X_train, y_train, X_test, y_test, plotting = False)

		ica_acc.append( acc)
		ica_train.append( train)
		ica_test.append(test)
		ica_iter.append( iterations)

		data_rp = GaussianRandomProjection(n_components = i).fit_transform(x)
		X_train, X_test, y_train, y_test = train_test_split(data_rp, y, test_size=0.2, random_state=42)
		train, test, acc, iterations =neural_network( X_train, y_train, X_test, y_test, plotting = False)

		rp_acc.append( acc)
		rp_train.append( train)
		rp_test.append(test)
		rp_iter.append( iterations)

		data_k_best = SelectKBest(chi2, k=i).fit_transform(x1, y)
		X_train, X_test, y_train, y_test = train_test_split(data_k_best, y, test_size=0.2, random_state=42)
		train, test, acc, iterations =neural_network( X_train, y_train, X_test, y_test, plotting = False)

		k_best_acc.append( acc)
		k_best_train.append( train)
		k_best_test.append(test)
		k_best_iter.append( iterations)


		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
		train, test, acc, iterations =neural_network( X_train, y_train, X_test, y_test, plotting = False)

		orig_acc.append( acc)
		orig_train.append( train)
		orig_test.append( test)
		orig_iter.append( iterations)



	plt.figure()


	plt.plot(np.arange(1, x.shape[1]),pca_iter, label = "PCA iterations ")
	plt.plot(np.arange(1, x.shape[1]),ica_iter,label = "ICA iterations ")
	plt.plot(np.arange(1, x.shape[1]),rp_iter,label = "RP iterations ")
	plt.plot(np.arange(1, x.shape[1]),k_best_iter,label = "Kbest iterations")
	plt.plot(np.arange(1, x.shape[1]),orig_iter,label = "Original iterations ")
	plt.legend()
	plt.title('iterations   vs. Number of components ')
	plt.xlabel('Number of Components')
	plt.ylabel('iterations ')
	plt.grid()
	plt.show()

	




	plt.figure()


	plt.plot(np.arange(1, x.shape[1]),pca_acc, label = "PCA accuracy")
	plt.plot(np.arange(1, x.shape[1]),ica_acc,label = "ICA accuracy")
	plt.plot(np.arange(1, x.shape[1]),rp_acc,label = "RP accuracy")
	plt.plot(np.arange(1, x.shape[1]),k_best_acc,label = "Kbest accuracy")
	plt.plot(np.arange(1, x.shape[1]),orig_acc,label = "Original accuracy")
	plt.legend()
	plt.title('Accuracy  vs. Number of components ')
	plt.xlabel('Number of Components')
	plt.ylabel('Accuracy')
	plt.grid()
	plt.show()


	plt.figure()


	plt.plot(np.arange(1, x.shape[1]),pca_train, label = "PCA train time")
	plt.plot(np.arange(1, x.shape[1]),ica_train,label = "ICA train time")
	plt.plot(np.arange(1, x.shape[1]),rp_train,label = "RP train time")
	plt.plot(np.arange(1, x.shape[1]),k_best_train,label = "Kbest train time")
	plt.plot(np.arange(1, x.shape[1]),orig_train,label = "Original train time")
	plt.legend()
	plt.title('training time  vs. Number of components ')
	plt.xlabel('Number of Components')
	plt.ylabel('train time ')
	plt.grid()
	plt.show()


	plt.figure()


	plt.plot(np.arange(1, x.shape[1]),pca_test, label = "PCA test time")
	plt.plot(np.arange(1, x.shape[1]),ica_test,label = "ICA test time")
	plt.plot(np.arange(1, x.shape[1]),rp_test,label = "RP test time")
	plt.plot(np.arange(1, x.shape[1]),k_best_test,label = "Kbest test time")
	plt.plot(np.arange(1, x.shape[1]),orig_test,label = "Original testing time")
	plt.legend()
	plt.title('testing time  vs. Number of components ')
	plt.xlabel('Number of Components')
	plt.ylabel('testing')
	plt.grid()
	plt.show()






def neural_network_analysis_kmeans( x,y):

	x1= x
	scaler=preprocessing.StandardScaler()#instantiate
	scaler.fit(x) 
	x=scaler.transform(x)
	
	kmeans_acc = []
	kmeans_train = []
	kmeans_test = []
	kmeans_iter = []

	n_cluster = 15
	for i in range( 1, n_cluster):
		cluster = KMeans( n_clusters =i, random_state=14, init= 'random')
		cluster_labels = cluster.fit(x)
		#x = np.array( x)
		#a = np.zeros(( 569, 2), dtype = np.float64)

		new_feature=  cluster_labels.predict(x)
		#x = np.hstack((x,[c])) 
		c = np.array(new_feature, copy=False, subok=True, ndmin=2).T
		x = np.append( x, c, axis =1)

		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
		train, test, acc, iterations =neural_network( X_train, y_train, X_test, y_test, plotting = False)
		kmeans_acc.append( acc)
		kmeans_train.append( train)
		kmeans_test.append(test)
		kmeans_iter.append(iterations)

		


	plt.figure()


	plt.plot(np.arange(1, n_cluster),kmeans_acc, label = "Kmeans accuracy")
	plt.legend()
	plt.title('Accuracy  vs. Number of clusters  ')
	plt.xlabel('Number of Clusters ')
	plt.ylabel('Accuracy')
	plt.grid()
	plt.show()


	plt.figure()


	plt.plot(np.arange(1, n_cluster),kmeans_train, label = "Kmeans train time")
	plt.legend()
	plt.title('training time  vs. Number of clusters  ')
	plt.xlabel('Number of Clusters ')
	plt.ylabel('train time ')
	plt.grid()
	plt.show()


	plt.figure()
	plt.plot(np.arange(1, n_cluster),kmeans_test, label = "Kmeans test time")
	plt.legend()
	plt.title('testing time  vs. Number of clusters  ')
	plt.xlabel('Number of Clusters ')
	plt.ylabel('testing')
	plt.grid()
	plt.show()


	plt.figure()
	plt.plot(np.arange(1, n_cluster),kmeans_iter, label = "Kmeans iterations ")
	plt.legend()
	plt.title('iterations  vs. Number of clusters ')
	plt.xlabel('Number of Clusters ')
	plt.ylabel('iterations ')
	plt.grid()
	plt.show()



def neural_network_analysis_EM( x,y):

	x1= x
	scaler=preprocessing.StandardScaler()#instantiate
	scaler.fit(x) 
	x=scaler.transform(x)
	
	em_acc1 = []
	em_train1 = []
	em_test1 = []
	em_iter1 = []

	n_cluster = 15
	for i in range( 1, n_cluster):
		gmm=mixture.GaussianMixture(i, n_init=50, random_state=14, covariance_type='spherical').fit(x) 
		labels=gmm.predict(x)
		c = np.array(labels, copy=False, subok=True, ndmin=2).T
		x = np.append( x, c, axis =1)

		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
		train, test, acc, iterations =neural_network( X_train, y_train, X_test, y_test, plotting = False)
		em_acc1.append( acc)
		em_train1.append( train)
		em_test1.append(test)
		em_iter1.append(iterations)


	em_acc2 = []
	em_train2 = []
	em_test2 = []
	em_iter2 = []

	n_cluster = 15
	for i in range( 1, n_cluster):
		gmm=mixture.GaussianMixture(i, n_init=50, random_state=14, covariance_type='tied').fit(x) 
		labels=gmm.predict(x)
		c = np.array(labels, copy=False, subok=True, ndmin=2).T
		x = np.append( x, c, axis =1)

		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
		train, test, acc, iterations =neural_network( X_train, y_train, X_test, y_test, plotting = False)
		em_acc2.append( acc)
		em_train2.append( train)
		em_test2.append(test)
		em_iter2.append(iterations)


	em_acc3 = []
	em_train3 = []
	em_test3 = []
	em_iter3 = []

	n_cluster = 15
	for i in range( 1, n_cluster):
		gmm=mixture.GaussianMixture(i, n_init=50, random_state=14, covariance_type='diag').fit(x) 
		labels=gmm.predict(x)
		c = np.array(labels, copy=False, subok=True, ndmin=2).T
		x = np.append( x, c, axis =1)

		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
		train, test, acc, iterations =neural_network( X_train, y_train, X_test, y_test, plotting = False)
		em_acc3.append( acc)
		em_train3.append( train)
		em_test3.append(test)
		em_iter3.append(iterations)


	em_acc4 = []
	em_train4 = []
	em_test4 = []
	em_iter4 = []

	n_cluster = 15
	for i in range( 1, n_cluster):
		gmm=mixture.GaussianMixture(i, n_init=50, random_state=14, covariance_type='full').fit(x) 
		labels=gmm.predict(x)
		c = np.array(labels, copy=False, subok=True, ndmin=2).T
		x = np.append( x, c, axis =1)

		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
		train, test, acc, iterations =neural_network( X_train, y_train, X_test, y_test, plotting = False)
		em_acc4.append( acc)
		em_train4.append( train)
		em_test4.append(test)
		em_iter4.append(iterations)


	
	
		


	plt.figure()


	plt.plot(np.arange(1, n_cluster),em_acc1, label = "covarince  spherical")
	plt.plot(np.arange(1, n_cluster),em_acc2, label = "covarince  tied")
	plt.plot(np.arange(1, n_cluster),em_acc3, label = "covarince  diag")
	plt.plot(np.arange(1, n_cluster),em_acc4, label = "covarince  full")
	plt.legend()
	plt.title('Accuracy  vs. Number of clusters  ')
	plt.xlabel('Number of Clusters ')
	plt.ylabel('Accuracy')
	plt.grid()
	plt.show()


	plt.figure()


	plt.plot(np.arange(1, n_cluster),em_train1, label = "covarince  spherical")
	plt.plot(np.arange(1, n_cluster),em_train2, label = "covarince  tied")
	plt.plot(np.arange(1, n_cluster),em_train3, label = "covarince  diag")
	plt.plot(np.arange(1, n_cluster),em_train4, label = "covarince  full")
	plt.legend()
	plt.title('training time  vs. Number of clusters  ')
	plt.xlabel('Number of Clusters ')
	plt.ylabel('train time ')
	plt.grid()
	plt.show()


	plt.figure()
	plt.plot(np.arange(1, n_cluster),em_test1, label = "covarince  spherical")
	plt.plot(np.arange(1, n_cluster),em_test2, label = "covarince  tied")
	plt.plot(np.arange(1, n_cluster),em_test3, label = "covarince  diag")
	plt.plot(np.arange(1, n_cluster),em_test4, label = "covarince  full")
	plt.legend()
	plt.title('testing time  vs. Number of clusters  ')
	plt.xlabel('Number of Clusters ')
	plt.ylabel('testing')
	plt.grid()
	plt.show()


	plt.figure()
	plt.plot(np.arange(1, n_cluster),em_iter1, label = "covarince  spherical")
	plt.plot(np.arange(1, n_cluster),em_iter2, label = "covarince  tied")
	plt.plot(np.arange(1, n_cluster),em_iter3, label = "covarince  diag")
	plt.plot(np.arange(1, n_cluster),em_iter4, label = "covarince  full")
	plt.legend()
	plt.title('iterations  vs. Number of clusters ')
	plt.xlabel('Number of Clusters ')
	plt.ylabel('iterations ')
	plt.grid()
	plt.show()



def nn_experiment5( x, y):
	scaler=preprocessing.StandardScaler()#instantiate
	scaler.fit(x) 
	x=scaler.transform(x)
	x1 =x 


	data_pca = PCA(n_components = 2).fit_transform(x)
	data_pca = FastICA(n_components = 15).fit_transform(x)

	data_rp = GaussianRandomProjection(n_components = 4).fit_transform(x)

	data_K_best = SelectKBest(chi2, k=i).fit_transform(x1, y)







def PCA_on_datasets( x, y,n):

	mod_data = PCA( n_components= n).fit_transform(x)
	k_means1( 2, x, y)
	EM( 6, mod_data, y, 'full')
	#pca_analysis_accuracy( x, y)
	#X_pca1 = pca_analysis( x)
	#plotting_data1(pca,  X_pca1, y)
	#X_pca2 = pca_analysis( X_train1)
	#plotting_data2( X_pca2, y_train1)
	#breakpoint()
	#mod_data1 = X_pca1[:, :2]
	#mod_data2 = X_pca2[:, :2]


	#pca_analysis_accuracy( X_train1, y_train1)


	#pca_analysis_accuracy( mod_data2, y_train1)



def RP_on_datasets( x, y,n):
	x = preprocessing.StandardScaler().fit_transform(x)
	#RP_analysis(x)
	#Z_2d = GaussianRandomProjection(n_components = 2).fit_transform(X_train1)
	#Plot_2d(Z_2d,y_train1)

	#Z_3d = GaussianRandomProjection(n_components = 3).fit_transform(X_train1)
	#Plot_3d(Z_3d,y_train1)
	best_n = n

	data_rp = GaussianRandomProjection(n_components = best_n).fit_transform(x)

	#k_means1( 2, data_rp, y)
	#EM( 2, data_rp, y, 'full')
	EM( 6, data_rp, y, 'full')


	#clustering_analysis_after_dimension( data_rp, y)




def K_best_on_dataset( x, y, n):
	x = preprocessing.MinMaxScaler().fit_transform(x)
	### Kbest ### 

	#reconstruction_error_K_best( x, y, 30)

	#reconstruction_error_K_best( X_train1, y_train1, 561)

	#x1 = preprocessing.MinMaxScaler().fit_transform(x)

	#data_k_best = SelectKBest(chi2, k=2).fit_transform(x1, y)

	#Plot_2d(data_k_best,y)

	best_n = n

	

	data_k_best = SelectKBest(chi2, k=best_n).fit_transform(x, y)
	#k_means1( 2, data_k_best, y)

	EM( 6, data_k_best, y, 'full')



def ICA_on_datasets( x, y, n):
	### ICA ###
	#kurtosis_analysis(x)
	#Z_2d = FastICA(n_components = 2).fit_transform(X_train1)
	#Plot_2d(Z_2d,y_train1)
	best_n = n

	mod_data = FastICA(n_components = best_n).fit_transform(x)
	#k_means1( 2, x, y)
	EM( 6, mod_data, y, 'full')

	#Plot_3d(Z_3d,y)









mod_data =  PCA( n_components = 60).fit_transform(X_train1)

breakpoint()
PCA_on_datasets( X_train1, y_train1, 65)
ICA_on_datasets( mod_data, y_train1, 10)
K_best_on_dataset( X_train1, y_train1, 530)
RP_on_datasets( X_train1, y_train1, 475)



breakpoint()
kurtosis_analysis_new1( mod_data, y, 6)

breakpoint()
k_means1( 2, mod_data, y)
EM( 2, x, y,'full')
EM( 2, mod_data, y, 'full')
gmm_analysis( mod_data, y)
breakpoint()

k_means( 2, mod_data, y)
breakpoint()
mod_data1 =  PCA( n_components = 65).fit_transform(X_train1)
EM( 6, mod_data1, y_train1,'full')
breakpoint()

gmm_analysis( mod_data1, y_train1)
k_means( 2, X_train1, y_train1)
breakpoint()

EM( 2, mod_data, y,'full')

k_means( 2, x, y)
k_means( 2, mod_data, y)
breakpoint()
EM( 2, x, y, 'spherical')
EM( 2, x, y, 'tied')
EM( 2, x, y, 'diag')
EM( 2, x, y, 'full')
mod_data1 =  PCA( n_components = 5).fit_transform(x)

breakpoint()
kurtosis_analysis_new1( mod_data1, y,5)

ica = FastICA(max_iter = 500)

S = ica.fit_transform(mod_data)
clustering_analysis_after_dimension( S, y_train1)
#reconstruction_error_pca(X_train1)

#neural_network_analysis_EM( x, y)

#neural_network_analysis_kmeans( x, y)
#neural_network_analysis( x,y)
#RP_analysis( X_train1)
#kurtosis_analysis(x)
#reconstruction_error_ica(x)
#pca_analysis(x)
#EM_accuracy( x, y)
#EM_accuracy( X_train1, y_train1)









best_n = 5
#pca_analysis_accuracy(x,y)

#mod_data1_ica = FastICA(n_components = best_n).fit_transform(x)
#Kmeans_clustering(mod_data1_ica )
#Kmeans_silhouette_analysis( mod_data1_ica, y)
#pca_analysis_accuracy( mod_data1_ica, y)
#k_means(  2, mod_data1_ica, y)
#gmm_cluster = gmm_analysis( mod_data1_ica,y)
#gmm_using_silhouette(mod_data1_ica, y)













### ICA  Data set 2###

#kurtosis_analysis(X_train1, 150)
#reconstruction_error_ica(X_train1, 100)
#Z_2d = FastICA(n_components = 2).fit_transform(X_train1)
#Plot_2d(Z_2d,y_train1)

#Z_3d = FastICA(n_components = 3).fit_transform(X_train1)
#plotting_data2( Z_3d, y_train1)

#Plot_3d(Z_3d,y_train1)
#best_n = 3

#mod_data1_ica = FastICA(n_components = best_n).fit_transform(X_train1)
#clustering_analysis_after_dimension(mod_data1_ica, y_train1 )
#Kmeans_clustering(mod_data1_ica )
#Kmeans_silhouette_analysis( mod_data1_ica, y_train1)
#pca_analysis_accuracy( mod_data1_ica, y_train1)
#gmm_cluster = gmm_analysis( mod_data1_ica,y_train1)
#gmm_using_silhouette(mod_data1_ica, y_train1)

















