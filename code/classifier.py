import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
from sklearn import tree, svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt 
import graph_tools as gt 
from sklearn.svm import LinearSVC
import pdb

debug = True

#f = open('../DataSet/train/X_train.txt')




def decision_tree( train_data, train_labels, test_data, test_labels, plotting):
    breakpoint()
    tree_clf = tree.DecisionTreeClassifier(random_state=34)
    tree_clf1=tree_clf # classifier used for plotting the learning and validation curve
    st=time.time()
    tree_clf.fit(train_data, train_labels)
    et=time.time()
    train_time= et-st
    st= time.time()
    predicted_labels = tree_clf.predict(test_data)
    et=time.time()
    testing_time= et-st
    gt.decision_tree_structure( train_data, train_labels, test_data, test_labels, tree_clf)
    accuracy = accuracy_score( test_labels, predicted_labels)
    if plotting: 
        gt.plot_learning_curve( train_data =train_data, train_labels=train_labels, clf =tree_clf1, clf_name="Decison tree", title="Learning curve",ylim=(0.7,1.01))
        gt.plot_validation_curve( train_data, train_labels, tree_clf1, "decision")
        matrix = confusion_matrix(test_labels, predicted_labels)
        gt.plot_confusion_matrix( matrix, class_names=None )
    if debug:
        print( 'the accuracy of the decision tree  is {}'.format(accuracy_score(test_labels, predicted_labels)))
        print(' the training time of decision tree is {}'.format(train_time))
        print( 'the testing time of decison tree is {}'.format(testing_time))

    
    return (train_time, testing_time, accuracy)






def optimized_decision_tree( train_data, train_labels, test_data, test_labels, plotting):
    clf = tree.DecisionTreeClassifier( random_state =34)
    
    param_range=[1,2,3,4,5,6,7,8,9,10,11,12,14]
    clf=GridSearchCV(clf, param_grid= {'max_depth': param_range},cv=5)
    st= time.time()
    clf.fit(train_data, train_labels)
    et=time.time()
    train_time=et-st
    st=time.time()
    predicted_labels=clf.predict(test_data)
    et=time.time()
    testing_time= et-st
    accuracy = accuracy_score( test_labels, predicted_labels)
    if plotting:
        clf1 = clf.best_estimator_
        matrix=confusion_matrix( test_labels, predicted_labels)
        gt.plot_learning_curve( train_data, train_labels, clf, 'decision tree', 'learning curve',cv=5, n_jobs=4, ylim=(0.7,1.01))
        gt.plot_confusion_matrix(matrix, class_names=None)
    
    if debug:
        print( 'the accuracy of optimized decision tree  is {}'.format(accuracy_score(test_labels, predicted_labels)))
        print(' the training time of decision tree is {}'.format(train_time))
        print( 'the testing time of decison tree is {}'.format(testing_time))
    return train_time, testing_time, accuracy




    



def svm_classifier( train_data, train_labels, test_data, test_labels, plotting,kernel ='linear',gamma='scale', C=0.01, degree=3):
    clf = svm.SVC( kernel = kernel, C=C, random_state=34, cache_size=500, gamma = gamma)
    #clf = Pipeline([
    #    ("scaler", StandardScaler()),
    #    ("svm_clf", svm.SVC(kernel="poly", degree=5, coef0=1, C=1))
    #])
    st = time.time()
    clf1=clf 
    clf.fit( train_data, train_labels)
    et = time.time() 
    train_time = et-st 
    st = time.time()
    predicted_labels = clf.predict( test_data)
    et = time.time()
    testing_time = et-st 
    title = "Learning Curve using SVM"
    clf_name ="svm"
    accuracy = accuracy_score( test_labels, predicted_labels)
    if( plotting == True):
        gt.plot_learning_curve( train_data, train_labels, clf1,clf_name, title, (0.7,1.01))
        gt.plot_validation_curve( train_data, train_labels, clf1, clf_name)
        matrix = confusion_matrix( test_labels, predicted_labels)
        gt.plot_confusion_matrix( matrix, class_names=None)

    if debug:
        print( 'The accuracy of svm classifier with {} kernel is {}'.format(kernel, accuracy_score( test_labels, predicted_labels)))
        print( 'the training time of {} is {}'.format( clf_name, train_time))
        print( 'the testing time of {} is {}'.format( clf_name, testing_time))
    return train_time, testing_time, accuracy



def optimized_svm_classifier( train_data, train_labels, test_data, test_labels, plotting, kernel='linear',gamma='scale', degree=3,C = 0.01):
    clf = svm.SVC( kernel = kernel, gamma=gamma, degree=degree, C=C, random_state=34, cache_size =500)
    ###breakpoint()

    

    C_range = [0.001,0.01,0.1,1,5,10,50,100]
    gamma_range = [0.001,0.01,.1, 1, 10]
    degree_range = [2,3,4,5,6,7,8,9]
    if( kernel == 'linear'):
        param_grid = dict( C=C_range)
    elif( kernel =='rbf'):
        param_grid = dict( gamma = gamma_range, C = C_range)
    else:
        param_grid =dict( degree = degree_range)
    clf = GridSearchCV(clf, param_grid, cv=5)
    st = time.time()
    clf.fit(train_data, train_labels)
    et = time.time() 
    train_time = et-st 
    st = time.time()
    predicted_labels = clf.predict( test_data)
    et = time.time()
    testing_time = et-st 
    title = "Learning Curve using SVM"
    clf_name ="svm"
    accuracy = accuracy_score( test_labels, predicted_labels)
    breakpoint()
    if( plotting == True):
        clf1 = clf.best_estimator_
        gt.plot_learning_curve( train_data, train_labels, clf1, clf_name, title,(0.7,1.01))
        matrix = confusion_matrix( test_labels, predicted_labels)
        gt.plot_confusion_matrix( matrix, class_names=None)
    
    if debug:
        print( 'The accuracy of optimized svm  classifier with {} kernel is {}'.format(kernel, accuracy_score( test_labels, predicted_labels)))
        print( 'the training time of {} is {}'.format( clf_name, train_time))
        print( 'the testing time of {} is {}'.format( clf_name, testing_time))
    return train_time, testing_time, accuracy



def knn_classifier( train_data, train_labels, test_data, test_labels, plotting,weighted =False):
    if weighted:
        clf  = KNeighborsClassifier( n_neighbors=1, weights='distance')
    else:
        clf=KNeighborsClassifier(n_neighbors=1)
    st = time.time()
    ##breakpoint()
    clf1=clf
    clf.fit( train_data, train_labels)
    et = time.time() 
    train_time = et-st 
    st = time.time()
    predicted_labels = clf.predict( test_data)
    et = time.time()
    testing_time = et-st 
    title = "Learning Curve using KNN"
    clf_name ="knn"
    accuracy = accuracy_score( test_labels, predicted_labels)
    if( plotting == True):
        gt.plot_learning_curve( train_data, train_labels,clf1, clf_name, title, ylim=(0.7,1.01))
        gt.plot_validation_curve( train_data, train_labels, clf1, 'knn')
        matrix = confusion_matrix( test_labels, predicted_labels)
        gt.plot_confusion_matrix( matrix, class_names=None)
    

    if debug:
        print( 'The accuracy of the knn classifier is {}'.format( accuracy_score( test_labels, predicted_labels)))
        print( 'the training time of {} is {}'.format( clf_name, train_time))
        print( 'the testing time of {} is {}'.format( clf_name, testing_time))
    return train_time, testing_time, accuracy


def optimized_knn_classifier( train_data, train_labels, test_data, test_labels, plotting,weighted=False):
    if weighted:
        clf  = KNeighborsClassifier( weights='distance', n_neighbors=3)
    else:
        clf=KNeighborsClassifier(n_neighbors=3)
    param_range= np.arange( 1, 8)
    clf= GridSearchCV(clf, param_grid={ 'n_neighbors': param_range}, cv=5)

    st = time.time()
    clf.fit( train_data, train_labels)
    ###breakpoint()
    et = time.time() 
    train_time = et-st 
    st = time.time()
    predicted_labels = clf.predict( test_data)
    et = time.time()
    testing_time = et-st 
    title = "Learning Curve using KNN"
    clf_name ="knn"
    accuracy = accuracy_score( test_labels, predicted_labels)
    if( plotting == True):
        clf1 = clf.best_estimator_

        gt.plot_learning_curve( train_data= train_data, train_labels= train_labels, clf = clf1, clf_name='knn', title='learning', ylim=(0.7,1.01))
        matrix = confusion_matrix( test_labels, predicted_labels)
        gt.plot_confusion_matrix( matrix, class_names=None)
    

    if debug:
        print( 'The accuracy of optimized knn classifier is {}'.format( accuracy_score( test_labels, predicted_labels)))
        print( 'the training time of {} is {}'.format( clf_name, train_time))
        print( 'the testing time of {} is {}'.format( clf_name, testing_time))
    return train_time, testing_time, accuracy



def GradientBoosting_classifier(train_data, train_labels, test_data, test_labels, plotting):
    boost_clf = GradientBoostingClassifier(max_depth =1, random_state=34, n_estimators=10, learning_rate=1)
    boost_clf1=boost_clf
    st=time.time()
    boost_clf.fit( train_data, train_labels)
    et= time.time()
    train_time= et-st
    st = time.time()
    predicted_labels= boost_clf.predict( test_data)
    et=time.time()
    testing_time= et-st
    clf_name='boosting'
    accuracy = accuracy_score( test_labels, predicted_labels)
    title="Learning Curve using Gradient Boosting"
    if plotting:
        matrix= confusion_matrix(test_labels, predicted_labels)
        gt.plot_learning_curve( train_data, train_labels, boost_clf1, 'boost decision', title, (0.7, 1.01), n_jobs=4)
        gt.plot_validation_curve( train_data, train_labels, boost_clf1, clf_name)
        gt.plot_confusion_matrix( matrix, class_names=None)
    
    if debug:
        print( 'The accuracy of the Adaboost  classifier is {}'.format( accuracy_score( test_labels, predicted_labels)))
        print( 'the training time of {} is {}'.format( clf_name, train_time))
        print( 'the testing time of {} is {}'.format( clf_name, testing_time))
    return train_time, testing_time, accuracy


def optimized_Gradient_Boosting_classifier(train_data, train_labels, test_data, test_labels, plotting):
    boost_clf = GradientBoostingClassifier(max_depth =1,random_state=34, n_estimators=10, learning_rate=0.1)
    estimator_range=np.arange( 10, 101,10 )
    max_depth_range = [1,2,3,4,5,6,7,8,9,10,11]
    param_grid =dict( max_depth = max_depth_range, n_estimators = estimator_range)

    boost_clf=GridSearchCV(boost_clf, param_grid, cv=5)
    st=time.time()
    boost_clf.fit( train_data, train_labels)
    et= time.time()
    train_time= et-st
    st = time.time()

    predicted_labels= boost_clf.predict( test_data)
    best_boost = boost_clf.best_estimator_;
    best_boost.fit( train_data, train_labels)

    et=time.time()
    testing_time= et-st
    clf_name='boosting'
    accuracy = accuracy_score( test_labels, predicted_labels)
    accuracy2 = accuracy_score( test_labels, best_boost.predict( test_data))
    title="Learning Curve using Gradient Boosting"
    

    if plotting:
        boost_clf1 = boost_clf.best_estimator_
        matrix= confusion_matrix(test_labels, predicted_labels)
        gt.plot_learning_curve( train_data, train_labels, boost_clf, 'boost decision', title, (0.7, 1.01), n_jobs=4)
        #gt.plot_validation_curve( train_data, train_labels, boost_clf1, clf_name)
        gt.plot_confusion_matrix( matrix, class_names=None)
    

    if debug:
        print( 'The accuracy of optimized adaboost classifier is {}'.format( accuracy_score( test_labels, predicted_labels)))
        print( 'the training time of {} is {}'.format( clf_name, train_time))
        print( 'the testing time of {} is {}'.format( clf_name, testing_time))
    return train_time, testing_time, accuracy



def neural_network( train_data, train_labels, test_data, test_labels,plotting, lr=1e-02, alpha=1):

    #gt.neural_network_hidden_layers( train_data, train_labels, test_data, test_labels)
    clf= MLPClassifier( activation='relu',max_iter=5000, alpha=alpha, batch_size='auto', learning_rate= 'adaptive',solver='adam',
        learning_rate_init=lr, random_state=34)
    clf1=clf;
    st=time.time()
    clf.fit( train_data, train_labels)
    et= time.time()
    train_time= et-st
    st = time.time()
    predicted_labels= clf.predict( test_data)
    et=time.time()
    testing_time= et-st
    clf_name='neural_network'
    accuracy = accuracy_score( test_labels, predicted_labels)
    title='Learning Curve with Neural Network'
    breakpoint()
    

    if plotting:
        matrix= confusion_matrix(test_labels, predicted_labels)
        gt.plot_learning_curve( train_data, train_labels, clf1, 'Neural network', title, (0.7, 1.01), n_jobs=4)
        gt.plot_validation_curve( train_data, train_labels, clf1, clf_name)
        gt.plot_validation_curve( train_data, train_labels, clf1, clf_name, 2)
        gt.plot_confusion_matrix( matrix, class_names=None)
        plt.plot(clf.loss_curve_)
        plt.title('Training loss curve for neural network')
        plt.xlabel('iterations')
        plt.ylabel("Loss")
        plt.grid()
        plt.show()
    
    if debug:
        print( 'The accuracy of the neural network classifier is {}'.format( accuracy_score( test_labels, predicted_labels)))
        print( 'the training time of {} is {}'.format( clf_name, train_time))
        print( 'the testing time of {} is {}'.format( clf_name, testing_time))
    return train_time, testing_time, accuracy




def optimized_neural_network( train_data, train_labels, test_data, test_labels,plotting, lr=1e-04, alpha =1e-05 ):
    clf= MLPClassifier( activation='relu',max_iter=5000, alpha=0.1, batch_size='auto', learning_rate= 'adaptive',solver='adam',
        learning_rate_init=0.0001, random_state=34)
    alpha_range= [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 1.5]
    lr_range = [0.00001,0.0001,0.001,0.01,0.1,0.5,0.8,1]
    ar_range=['relu', 'tanh']
    params={ 'alpha': alpha_range, 'learning_rate_init':lr_range, 'activation': ar_range }
    clf = GridSearchCV( clf, param_grid= params, cv=5)

    st=time.time()
    clf1=clf
    clf.fit( train_data, train_labels)
    et= time.time()
    train_time= et-st
    st = time.time()
    predicted_labels= clf.predict( test_data)
    et=time.time()
    testing_time= et-st
    clf_name='neural_network'
    accuracy = accuracy_score( test_labels, predicted_labels)
    title='confusion matrix with Neural Network '
    
    if plotting:
        matrix= confusion_matrix(test_labels, predicted_labels)
        best_classifier = MLPClassifier( activation = 'relu', alpha=clf.best_params_['alpha'], 
            learning_rate_init=clf.best_params_['learning_rate_init'], random_state=42,solver='adam' )
        gt.plot_learning_curve( train_data, train_labels, clf1, 'Neural network', title, (0.7, 1.01), n_jobs=4)
        gt.plot_confusion_matrix( matrix, class_names=None)
        #best_classifier = MLPClassifier( activation = clf.best_params_['activation'], alpha= clf.best_params_['alpha'], 
        #    learning_rate_init=clf.best_params_['learning_rate_init'],hidden_layer_sizes=(5,2), max_iter=1, random_state=42 )
        best_classifier = MLPClassifier( activation = 'relu', alpha= 0.1, 
            learning_rate_init=0.0001,hidden_layer_sizes=(5,2), max_iter=1, random_state=42,solver='lbfgs' )
        gt.neural_networks_graph( train_data, train_labels, test_data, test_labels, clf, 300)
    
    if debug:
        print( 'The accuracy of the Neural network classifier is {}'.format( accuracy_score( test_labels, predicted_labels)))
        print( 'the training time of {} is {}'.format( clf_name, train_time))
        print( 'the testing time of {} is {}'.format( clf_name, testing_time))
    return train_time, testing_time, accuracy







#knn_classifier( data_set, new_labels, test_data[:50], test_labels[:50], True)

#decision_tree( data_set, new_labels, test_data, test_labels, False)
#optimized_decision_tree(data_set, new_labels, test_data, test_labels, False)



