import numpy as np 
from sklearn import preprocessing 
#import classifier as ct
import pdb
from sklearn import tree, svm
#import graph_tools as gt
import pandas as pd
import matplotlib.pyplot as plt 



def data_set( train_data, train_labels, test_data, test_labels):
    return train_data, train_labels, test_data, test_labels




def load_data( prefix, file_name, suffix):
    values = list()
    for names in file_name:
        data=pd.read_csv( prefix+names+suffix, header=None, delim_whitespace=True)
        values.append(data)
    data = np.dstack( values)
    return data



def data_for_one_subject(X_train, y_train, sub_list, subid):
    ix = [i for i in range(len(sub_list)) if sub_list[i] == subid]
    return X_train[ix,:,:], y_train[ix]


def remove_overlap( windows):
    series  = list()
    for x in windows:
        #print( x.shape)
        half = int( len( windows)/2) -1
        for value in x[-half:]:
            series.append(value)
    return series


def plot_data_for_one_subject( sub_X, sub_y):
    n_plots, off = sub_X.shape[2]+1, 0
    plt.figure( figsize=(15,15))
    #plot total_acc 
    for i in range(3):
        plt.subplot( n_plots, 1, off+1)
        plt.plot( remove_overlap( sub_X[:,:,off]),'b')
        plt.title('total acc '+str(i), y=0, loc='left')
        off+=1
    for i in range(3):
        plt.subplot( n_plots, 1, off+1)
        plt.plot( remove_overlap( sub_X[:,:,off]),'g')
        plt.title('body acc '+str(i), y=0, loc='left')
        off+=1
    for i in range(3):
        plt.subplot( n_plots, 1, off+1)
        plt.plot( remove_overlap( sub_X[:,:,off]),'r')
        plt.title('body gyro '+str(i), y=0, loc='left')
        off+=1
    plt.subplot( n_plots, 1, n_plots)
    plt.plot(sub_y,'k')
    plt.title('activity', y=0, loc='left')
    plt.show()
    


def extract_and_plot_movement_data( ):
    prefix_train = '../DataSet/train/Inertial Signals/'
    prefix_test = '../DataSet/test/Inertial Signals/'
    body_acc = [ 'total_acc_x_','total_acc_y_','total_acc_z_']
    body_gyro = [ 'body_gyro_x_','body_gyro_y_','body_gyro_z_']
    total_acc = ['total_acc_x_','total_acc_y_','total_acc_z_']
    total_files = body_acc + body_gyro+ total_acc
    X_train = load_data(prefix_train, total_files,'train.txt')
    y_train = pd.read_csv( '../DataSet/train/y_train.txt', header=None, delim_whitespace=True).values
    X_test = load_data( prefix_test, total_files,'test.txt')
    y_test =  pd.read_csv( '../DataSet/test/y_test.txt', header=None, delim_whitespace=True).values
    subjects = pd.read_csv('../DataSet/train/subject_train.txt', header=None, delim_whitespace=True).values
    sub_X, sub_y = data_for_one_subject( X_train, y_train, subjects, 1)
    plot_data_for_one_subject( sub_X, sub_y)


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
X_train1 = preprocessing.scale(data_set)
y_train1 = extract_labels('../DataSet/train/y_train.txt')
X_test1 = prepare_dataSet('../DataSet/test/X_test.txt')
X_test1= preprocessing.scale( test_data)

y_test1 = extract_labels('../DataSet/test/y_test.txt')


results = list()

plotting = False
def main():
    breakpoint()

    #ct.neural_network(data_set, new_labels, test_data, test_labels, plotting )
    #plotting the input data 
    #gt.plot_input_data( new_labels)



    # KNN classifier and its optimized version 

    #ct.knn_classifier( data_set, new_labels, test_data, test_labels, plotting)
    #ct.optimized_knn_classifier( data_set, new_labels, test_data, test_labels, plotting, weighted=False)
    
    
    #decison tree and its optimized version
    #ct.decision_tree( data_set, new_labels, test_data, test_labels, plotting)
    #ct.optimized_decision_tree( data_set, new_labels, test_data, test_labels, plotting)
    

    # SVM classifier and its optimized version
    #ct.svm_classifier( data_set, new_labels, test_data, test_labels, plotting,'linear',C=0.25)
    #ct.svm_classifier( data_set, new_labels, test_data, test_labels, plotting,'rbf',C=10, gamma =0.001)
    #ct.svm_classifier( data_set, new_labels, test_data, test_labels, plotting,'poly',degree=3)
    #x= ct.optimized_svm_classifier( data_set, new_labels, test_data, test_labels, plotting,kernel ='linear', C=0.25)
    #y= ct.optimized_svm_classifier( data_set, new_labels, test_data, test_labels, plotting,kernel='rbf',gamma=0.001,C= 10 )
    #z= ct.optimized_svm_classifier( data_set, new_labels, test_data, test_labels, plotting,kernel ='poly', degree=3)
    #ct.optimized_svm_classifier( data_set, new_labels, test_data, test_labels, True, 'rbf')

    #ct.optimized_svm_classifier( data_set, new_labels, test_data, test_labels, True, 'poly')
    #ct.GradientBoosting_classifier( data_set, new_labels, test_data, test_labels, plotting )
    #ct.optimized_neural_network( data_set, new_labels, test_data, test_labels, True)


if __name__ == '__main__':
    main()






