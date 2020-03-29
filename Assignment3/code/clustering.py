#import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import os
from sklearn.decomposition import PCA as PCA_sk
from sklearn.decomposition import FastICA, TruncatedSVD
from sklearn.decomposition import NMF as NMF_sk
from numpy import linalg as LA
import scipy
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, silhouette_samples
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.metrics import homogeneity_score, completeness_score, \
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn import mixture
import itertools
from scipy import linalg
from sklearn import metrics

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
os.system('clear')



data =datasets.load_breast_cancer()
x, y = data.data, data.target 

x = preprocessing.StandardScaler().fit_transform(x)


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
X_train1 = preprocessing.StandardScaler().fit_transform(X_train1)
#X_train1 = preprocessing.scale(X_train1)
y_train1 = extract_labels('../DataSet/train/y_train.txt')
X_test1 = prepare_dataSet('../DataSet/test/X_test.txt')
X_test1= preprocessing.scale( X_test1)

y_test1 = extract_labels('../DataSet/test/y_test.txt')





def EM(n_clust, data_frame, true_labels, variance):
    gmm=mixture.GaussianMixture(n_clust, n_init=20, covariance_type = variance).fit(data_frame) 
    c_labels=gmm.predict(data_frame)
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    #y_clust = k_means.predict(data_frame)
    breakpoint()
    print( ct)
    print( "SA score is " , silhouette_score(data_frame, c_labels, metric='euclidean'))




	



def k_means1(n_clust, data_frame, true_labels):

    for i in range( 2, 3):
    	k_means = KMeans(n_clusters = i, random_state=14, n_init = 50)
    	k_means.fit(data_frame)
    	c_labels = k_means.labels_
    	y_clust = k_means.predict(data_frame)
    	df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    	ct = pd.crosstab(df['clust_label'], df['orig_label'])



    	
    	print( ct)
    	print( " Silhouette score ", silhouette_score(data_frame, y_clust, metric='euclidean'))
    	#print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    	#print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'%(k_means.inertia_,
    	#	homogeneity_score(true_labels, y_clust),completeness_score(true_labels, y_clust),
    	#	v_measure_score(true_labels, y_clust),adjusted_rand_score(true_labels, y_clust),
    	#	adjusted_mutual_info_score(true_labels, y_clust),
    	#	silhouette_score(data_frame, y_clust, metric='euclidean')))


def k_means(n_clust, data_frame, true_labels):

    for i in range( 2, 10):
    	k_means = KMeans(n_clusters = i, random_state=14, n_init = 50)
    	k_means.fit(data_frame)
    	c_labels = k_means.labels_
    	df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    	ct = pd.crosstab(df['clust_label'], df['orig_label'])
    	y_clust = k_means.predict(data_frame)
    	print( ct)
    	print( " Silhouette score ", silhouette_score(data_frame, y_clust, metric='euclidean'))

def Kmeans_clustering( X):
	clusters = list(range( 1, 10))
	inertia_arr = []

	for c in clusters:
		print(c)
		cluster = KMeans( n_clusters =c, random_state=14, n_init = 50)
		cluster_labels = cluster.fit(X)
		inertia_arr.append( cluster.inertia_)
	inertia_arr =np.array( inertia_arr)
	plt.plot(clusters,inertia_arr)
	plt.xlabel('Number of Clusters')
	plt.ylabel('Inertia')
	plt.title('Choosing Best k with Inertia')
	plt.grid()
	plt.show()



def Kmeans_silhouette_analysis(X,y):
	cluster_range = [2,3,4, 5,6,7,8,9,10,11,12]
	for num_cluster in cluster_range:
	    fig, (ax1, ax2) = plt.subplots(1, 2)
	    fig.set_size_inches(18, 7)
	   
	    ax1.set_xlim([-0.1, 1])
	    ax1.set_ylim([0, len(X) + (num_cluster + 1) * 10])

	    clusterer = KMeans(n_clusters=num_cluster, random_state=10)
	    cluster_labels = clusterer.fit_predict(X)

	    silhouette_avg = silhouette_score(X, cluster_labels)
	    print("For n_clusters = ", num_cluster,
	          "The average silhouette_score is :", silhouette_avg)
	    sample_silhouette_values = silhouette_samples(X, cluster_labels)

	    y_lower = 10
	    for i in range(num_cluster):
	        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

	        ith_cluster_silhouette_values.sort()

	        size_cluster_i = ith_cluster_silhouette_values.shape[0]
	        y_upper = y_lower + size_cluster_i

	        color = cm.nipy_spectral(float(i) / num_cluster)
	        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

	        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

	        y_lower = y_upper + 10 

	    ax1.set_title("The silhouette plot for the various clusters.")
	    ax1.set_xlabel("The silhouette coefficient values")
	    ax1.set_ylabel("Cluster label")

	    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

	    ax1.set_yticks([])  
	    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

	    colors = cm.nipy_spectral(cluster_labels.astype(float) / num_cluster)
	    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
	                c=colors, edgecolor='k')

	    centers = clusterer.cluster_centers_

	    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')

	    for i, c in enumerate(centers):
	        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

	    ax2.set_title("The visualization of the clustered data.")
	    ax2.set_xlabel("Feature space for the 1st feature")
	    ax2.set_ylabel("Feature space for the 2nd feature")

	    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data with num_cluster = %d" % num_cluster),fontsize=14, fontweight='bold')
	plt.show()



def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]

def gmm_using_silhouette(X, labels):
	n_clusters=np.arange(2, 10)
	sils=[]
	sils_err=[]
	iterations=20
	for n in n_clusters:
		tmp_sil=[]
		for _ in range(iterations):
			gmm=mixture.GaussianMixture(n, n_init=2).fit(X) 
			labels=gmm.predict(X)
			sil=metrics.silhouette_score(X, labels, metric='euclidean')
			tmp_sil.append(sil)
		val=np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
		err=np.std(tmp_sil)
		sils.append(val)
		sils_err.append(err)
	plt.errorbar(n_clusters, sils, yerr=sils_err)
	plt.title("Silhouette Scores", fontsize=20)
	plt.xticks(n_clusters)
	plt.xlabel("N. of clusters")
	plt.ylabel("Score")
	plt.grid()
	plt.show()



def gmm_analysis(X,y):
	lowest_bic = np.infty
	bic = []
	n_components_range = range(1, 10)
	cv_types = ['spherical', 'tied', 'diag', 'full']
	for cv_type in cv_types:
	    for n_components in n_components_range:
	        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, max_iter = 200)
	        gmm.fit(X)
	        bic.append(gmm.bic(X))
	        if bic[-1] < lowest_bic:
	            lowest_bic = bic[-1]
	            best_gmm = gmm

	bic = np.array(bic)
	color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue','darkorange'])
	clf = best_gmm
	bars = []
	plt.figure(figsize=(8, 6))
	spl = plt.subplot(2, 1, 1)

	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
	    xpos = np.array(n_components_range) + .2 * (i - 2)
	    bars.append(plt.bar(xpos, bic[i * len(n_components_range):(i + 1) * len(n_components_range)], width=.2, color=color))
	
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bic.argmin() / len(n_components_range))
	best_num = np.mod(bic.argmin(), len(n_components_range)) + 1
	plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
	spl.set_xlabel('Number of components')
	spl.legend([b[0] for b in bars], cv_types)

	
	splot = plt.subplot(2, 1, 2)
	Y_ = clf.predict(X)
	
	for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,color_iter)):
	    v, w = linalg.eigh(cov)
	    if not np.any(Y_ == i):
	        continue
	    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

	    angle = np.arctan2(w[0][1], w[0][0])
	    angle = 180. * angle / np.pi 
	    v = 2. * np.sqrt(2.) * np.sqrt(v)
	    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
	    ell.set_clip_box(splot.bbox)
	    ell.set_alpha(.5)
	    splot.add_artist(ell)
	
	plt.xticks(())
	plt.yticks(())
	plt.title('Selected GMM: full model,' + str(best_num) + ' components')
	plt.subplots_adjust(hspace=.35, bottom=.02)
	plt.show()
	return best_num


#breakpoint()


#Kmeans_clustering(x)
#EM( 2, x, y)
#Kmeans_clustering(x)
#Kmeans_silhouette_analysis(x, y)
#k_means( 2, x, y)

#k_means( 2, X_train1, y_train1)
#gmm_cluster = gmm_analysis( X_train1,y_train1)
#gmm_using_silhouette(X_train1, y_train1)

#evaluate_cluster( gmm_cluster. )










