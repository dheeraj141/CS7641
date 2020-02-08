import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, datasets
import matplotlib.pyplot as plt
import classifier as ct
import graph_tools as gt 








data1 = datasets.load_breast_cancer()
x, y = data1.data, data1.target
print(x.shape,y.shape)
#breakpoint()

#gt.plot_input_data( y, ['Malignant' , 'Benign'], 'Histogram')

#preprocessing the data 
x=preprocessing.scale(x)

#splitting the data into training and testing set 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

classifiers=[ 'neural_network' ]
training_time = np.arange( len( classifiers),dtype=np.float)
testing_time = np.arange( len(classifiers),dtype=np.float)
training_time_tuned = np.arange( len( classifiers),dtype=np.float)
testing_time_tuned= np.arange( len( classifiers),dtype=np.float)


"""
breakpoint()
plotting=False
accuracy= np.arange( len( classifiers),dtype=np.float)
accuracy_tuned = np.arange( len( classifiers),dtype=np.float)
for index, clf in enumerate( classifiers):
	if clf == 'neural_network':
		plotting=True
	

	training_time[index], testing_time[index], accuracy[index] =call_classifier( X_train, y_train, X_test, y_test,clf, plotting,False)
	#training_time_tuned[index], testing_time_tuned[index], accuracy_tuned[index] = call_classifier( X_train, y_train, X_test, y_test,clf, False, True)


breakpoint()

for index,clf in enumerate( classifiers):
	print( 'The classifier is {} with accuracy {} and tuned accuracy {}'.format(clf, accuracy[index], accuracy_tuned[index]))

"""

""" main function to  run the classifier"""
plotting = False

def main(): 
	breakpoint()
	ct.GradientBoosting_classifier( X_train, y_train, X_test, y_test,plotting)
	#ct.optimized_Gradient_Boosting_classifier(X_train, y_train, X_test, y_test,plotting)
	#ct.neural_network( X_train, y_train, X_test, y_test,plotting)
	#ct.optimized_neural_network( X_train, y_train, X_test, y_test,plotting )

	#ct.decision_tree( X_train, y_train, X_test, y_test,plotting)
	#ct.optimized_decision_tree( X_train, y_train, X_test, y_test, plotting)
	#ct.knn_classifier(X_train, y_train, X_test, y_test, plotting )
	#ct.optimized_knn_classifier( X_train, y_train, X_test, y_test, plotting)
	#ct.svm_classifier( X_train, y_train, X_test, y_test, plotting,'rbf')
	#ct.optimized_svm_classifier(X_train, y_train, X_test, y_test,plotting,'linear' )
	#ct.optimized_svm_classifier(X_train, y_train, X_test, y_test,plotting,'rbf' )
	#ct.optimized_svm_classifier(X_train, y_train, X_test, y_test,plotting,'poly' )
    #x= ct.optimized_svm_classifier( data_set, new_labels, test_data, test_labels, False,kernel ='linear', C=0.25)
    #y= ct.optimized_svm_classifier( data_set, new_labels, test_data, test_labels, False,kernel='rbf',gamma=0.001,C= 10 )
    #z= ct.optimized_svm_classifier( data_set, new_labels, test_data, test_labels, False,kernel ='poly', degree=3)
    #ct.optimized_svm_classifier( data_set, new_labels, test_data, test_labels, True, 'rbf')

    #ct.optimized_svm_classifier( data_set, new_labels, test_data, test_labels, True, 'poly')






if __name__ == '__main__':
    main()

