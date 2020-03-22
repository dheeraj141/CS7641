import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
import itertools
import time
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier






def plot_input_data( y_train, names, title):
	breakpoint()
	#names = ['WALKING', 'WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
	values = np.bincount(y_train)
	#values = values[1:]

	#plt.figure(figsize=(10, 5))
	plt.bar(names, values)
	plt.title(title)
	#plt.figtext( 10,1,  "A ='WALKING B= WALKING_UPSTAIRS C =WALKING_DOWNSTAIRS D= SITTING STANDING LAYING" )
	plt.show()


""" 
to plot the influence of each hyperparameter on the learning accuracy and the validation accuracy
high training score and low validation score means overfitting otherwise it is fine

"""
def plot_validation_curve(train_data, train_labels, clf, clf_name, flag =1):
	if (clf_name == 'svm'):
		pm = 'C'
		pr = np.logspace(-2, 1,10)
	elif(clf_name == 'decision'):
		pm= 'max_depth'
		pr = np.arange(1,30)
	elif( clf_name == 'knn'):
		pr = np.arange(1, 9)
		pm = 'n_neighbors'
	elif(clf_name == 'neural_network'):
		if( flag ==1):
			pm = 'alpha'
			pr = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 50, 100]
		else:
			pm ='learning_rate_init'
			pr =[0.00001,0.0001,0.001,0.01,0.1]
	elif(clf_name == "boosting"):
		pr = np.arange(10, 200,10 )
		pm = 'n_estimators'
	else :
		print("wrong input ")
		return 
	#breakpoint()

	train_scores, test_scores = validation_curve(clf, train_data, train_labels, pm,pr,cv=5,scoring='accuracy', n_jobs=4)
	train_scores_mean=np.mean(train_scores, axis=1)
    #train_scores,test_scores=validation_curve(clf,train_data,train_labels,param_name=pm,pr=pr,cv=5,scoring="accuracy", n_jobs=1)
    #train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.title( "validation_curve for {}".format(clf))
	plt.xlabel(pm)
	plt.ylabel("score")
	plt.ylim(0.0, 1.1)
	plt.grid()
	lw = 2
	plt.plot(pr, train_scores_mean, label="Training score",color="darkorange", lw=lw)
	plt.fill_between(pr, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.2,color="darkorange", lw=lw)
	plt.plot(pr, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
	plt.fill_between(pr, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.2,color="navy", lw=lw)
	plt.legend(loc="best")
	plt.show()


""" this is used to plot the effect of the training size of the data on the validation accuracy"""


def plot_learning_curve( train_data, train_labels , clf, clf_name, title, ylim=None,cv=None, n_jobs=4,train_sizes = np.linspace( 0.1, 1, 5)):
	plt.figure()
	plt.title( title)
	plt.xlabel(" training size")
	plt.ylabel("accuracy")
	if ylim is not None:
		plt.ylim(*ylim)

	
	train_sizes , train_scores, test_scores = learning_curve( clf, train_data, train_labels , cv=cv, n_jobs=n_jobs, train_sizes = train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()
	plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std,alpha=0.1,color="r")
	plt.fill_between(train_sizes, test_scores_mean-test_scores_std, test_scores_mean+test_scores_std,alpha=0.1, color="g")
	plt.plot( train_sizes,train_scores_mean, 'o-',color="r", label="train_scores")
	plt.plot(train_sizes, test_scores_mean,'o-',color="g",label="Cross-validation score")
	plt.legend(loc="best")
	plt.show()
	return plt

def plot_confusion_matrix( matrix , class_names= None, title="confusion matrix", cmap=plt.get_cmap('Blues'),Normalize=True):
	accuracy=np.trace(matrix)/float(np.sum( matrix))
	misclass =1- accuracy
	plt.figure( figsize=(8,6))
	plt.imshow( matrix, interpolation='nearest', cmap=cmap )
	plt.title(title)
	plt.colorbar()

	if class_names is not None :
		tick_marks = np.arange(len( class_names))
		plt.xticks( tick_marks, class_names, rotation=45)
		plt.yticks( tick_marks, class_names)
	if Normalize:
		matrix=matrix.astype('float')/matrix.sum( axis=1)[:, np.newaxis]
	thresh = matrix.max()/1.5 if Normalize else matrix.max()/2
	for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
		if Normalize:
			plt.text( j,i, "{:0.4f}".format(matrix[i, j]),horizontalalignment="center",color="white" if matrix[i,j]> thresh else "black")
		else:
			plt.text( j,i ,"{:,}".format(matrix[i, j]),horizontalalignment="center", color="white" if matrix[i, j]> thresh else "black")
	plt.tight_layout()
	plt.ylabel('True Label')
	plt.xlabel( 'Predicted label\naccuracy= {:0.4f}; misclass={:0.4f}'.format( accuracy, misclass))
	plt.show()




def neural_networks_graph( train_data, train_labels, test_data, test_labels,clf, epochs ):
	breakpoint()
	train_score=np.zeros( epochs)
	train_loss = np.zeros( epochs)
	for i in range( epochs):
		clf.fit(train_data, train_labels)
		train_loss[i] = clf.loss_
		train_score[i] = accuracy_score( train_labels, clf.predict( train_data))

	range_loss= np.arange( epochs)+1
	plt.plot(range_loss, train_loss)
	plt.title('Training loss curve for neural network')
	plt.xlabel('Epochs')
	plt.ylabel("Loss")
	plt.grid()
	plt.show()

""" This function code is taken from the web and reference is pointed below"""


def decision_tree_structure( X_train, y_train, X_test, y_test,estimator):
	# Using those arrays, we can parse the tree structure:

	n_nodes = estimator.tree_.node_count
	children_left = estimator.tree_.children_left
	children_right = estimator.tree_.children_right
	feature = estimator.tree_.feature
	threshold = estimator.tree_.threshold


	# The tree structure can be traversed to compute various properties such
	# as the depth of each node and whether or not it is a leaf.
	node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
	is_leaves = np.zeros(shape=n_nodes, dtype=bool)
	stack = [(0, -1)]  # seed is the root node id and its parent depth
	while len(stack) > 0:
	    node_id, parent_depth = stack.pop()
	    node_depth[node_id] = parent_depth + 1

	    # If we have a test node
	    if (children_left[node_id] != children_right[node_id]):
	        stack.append((children_left[node_id], parent_depth + 1))
	        stack.append((children_right[node_id], parent_depth + 1))
	    else:
	        is_leaves[node_id] = True

	print("The binary tree structure has %s nodes and has "
	      "the following tree structure:"
	      % n_nodes)
	for i in range(n_nodes):
	    if is_leaves[i]:
	        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
	    else:
	        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
	              "node %s."
	              % (node_depth[i] * "\t",
	                 i,
	                 children_left[i],
	                 feature[i],
	                 threshold[i],
	                 children_right[i],
	                 ))
	print()

	# First let's retrieve the decision path of each sample. The decision_path
	# method allows to retrieve the node indicator functions. A non zero element of
	# indicator matrix at the position (i, j) indicates that the sample i goes
	# through the node j.

	node_indicator = estimator.decision_path(X_test)

	# Similarly, we can also have the leaves ids reached by each sample.

	leave_id = estimator.apply(X_test)

	# Now, it's possible to get the tests that were used to predict a sample or
	# a group of samples. First, let's make it for the sample.

	sample_id = 0
	node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
	                                    node_indicator.indptr[sample_id + 1]]

	print('Rules used to predict sample %s: ' % sample_id)
	for node_id in node_index:
	    if leave_id[sample_id] == node_id:
	        continue

	    if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
	        threshold_sign = "<="
	    else:
	        threshold_sign = ">"

	    print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
	          % (node_id,
	             sample_id,
	             feature[node_id],
	             X_test[sample_id, feature[node_id]],
	             threshold_sign,
	             threshold[node_id]))

	# For a group of samples, we have the following common node.
	sample_ids = [0, 1]
	common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
	                len(sample_ids))

	common_node_id = np.arange(n_nodes)[common_nodes]

	print("\nThe following samples %s share the node %s in the tree"
	      % (sample_ids, common_node_id))
	print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))




def neural_network_hidden_layers( train_data, train_labels, test_data, test_labels):
	accuracy = np.zeros( 10)
	train_time = np.zeros(10)
	test_time = np.zeros(10)
	for i in range( 5,15):
		clf= MLPClassifier( activation='relu',max_iter=1000, alpha=0.1, batch_size='auto', learning_rate= 'adaptive',solver='adam',learning_rate_init=0.01, hidden_layer_sizes= (10, i))
		st=time.time()
		clf.fit( train_data, train_labels)
		et = time.time()
		train_time[i-5] = et-st 
		st = time.time()
		predicted_labels = clf.predict( test_data)
		test_time[i-5] = time.time() - st 
		accuracy[i-5] = accuracy_score( test_labels, predicted_labels)

	breakpoint()
	print( train_time)
	print(test_time)
	print( accuracy)






















