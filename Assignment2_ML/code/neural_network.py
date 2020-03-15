import numpy as np 
import mlrose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, datasets
import time
from random import randint
import warnings



data =datasets.load_breast_cancer()
x, y = data.data, data.target 

x = preprocessing.scale(x)
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size =0.4, random_state =42)


def neural_network_iterations():

	rhc_train_acc = []
	sa_train_acc= []
	ga_train_acc = []
	mimic_train_acc = []
	rhc_test_acc =[]
	sa_test_acc= []
	ga_test_acc = []
	mimic_test_acc =[]
	rhc_time =[]
	sa_time =[]
	ga_time = []
	bd_time = []
	bd_train_acc=[]
	bd_test_acc=[]
	start =10
	end=1000
	step=20


	for iter in range( start, end, step):
		print( iter, " started ")

		rhc_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='random_hill_climb', max_iters = iter, bias = True, 
			is_classifier = True, learning_rate = 1e-03, early_stopping = True, max_attempts = 1000, random_state =42)

		st = time.time()
		rhc_model.fit( X_train, y_train)
		et = time.time()
		tt = et-st 

		#predict labels 

		y_predict = rhc_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		rhc_train_acc.append( train_acc)

		#testing accuracy 
		y_predict = rhc_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		rhc_test_acc.append( test_acc)
		rhc_time.append( tt)




		sa_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='simulated_annealing', max_iters = iter, bias = True, 
			is_classifier = True, learning_rate = 1e-03, early_stopping = True, max_attempts = 1000, random_state = 42)









		st = time.time()
		sa_model.fit( X_train, y_train)
		et = time.time()
		tt = et-st 

		#predict labels 

		y_predict = sa_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		sa_train_acc.append( train_acc)

		#testing accuracy 
		y_predict = sa_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		sa_test_acc.append( test_acc)
		sa_time.append( tt)



		ga_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='genetic_alg', max_iters = iter, bias = True, 
			is_classifier = True, learning_rate = 1e-03, early_stopping = True, max_attempts = 1000, random_state = 42)









		st = time.time()
		ga_model.fit( X_train, y_train)
		et = time.time()
		tt = et-st 

		#predict labels 

		y_predict = ga_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		ga_train_acc.append( train_acc)

		#testing accuracy 
		y_predict = ga_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		ga_test_acc.append( test_acc)
		ga_time.append( tt)


		bd_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='gradient_descent', max_iters = iter, bias = True, 
			is_classifier = True, learning_rate = 1e-03, early_stopping = True, max_attempts = 1000, random_state = 42)









		st = time.time()
		bd_model.fit( X_train, y_train)
		et = time.time()
		tt = et-st 

		#predict labels 

		y_predict = bd_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		bd_train_acc.append( train_acc)

		#testing accuracy 
		y_predict = bd_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		bd_test_acc.append( test_acc)
		bd_time.append( tt)
		print(iter,"done")


	plt.figure()
	plt.plot(np.arange(start, end, step),np.array(rhc_train_acc),label='RHC')
	plt.plot(np.arange(start, end, step),np.array(sa_train_acc),label='SA')
	plt.plot(np.arange(start, end, step),np.array(ga_train_acc),label='GA')
	plt.plot(np.arange(start, end, step),np.array(bd_train_acc),label='GD')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Train Accuracy')
	plt.title('Train Accuracy vs. Iterations for Optimization Algorithms')
	plt.legend()
	plt.savefig('nn_train_iterations.png')



	plt.figure()
	plt.plot(np.arange(start, end, step),np.array(rhc_test_acc),label='RHC')
	plt.plot(np.arange(start, end, step),np.array(sa_test_acc),label='SA')
	plt.plot(np.arange(start, end, step),np.array(ga_test_acc),label='GA')
	plt.plot(np.arange(start, end, step),np.array(bd_test_acc),label='GD')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Test Accuracy')
	plt.title('Test Accuracy vs. Iterations for Optimization Algorithms')
	plt.legend()
	plt.savefig('nn_test_iterations.png')


	plt.figure()
	plt.plot(np.arange(start, end, step),np.array(rhc_time),label='RHC')
	plt.plot(np.arange(start, end, step),np.array(sa_time),label='SA')
	plt.plot(np.arange(start, end, step),np.array(ga_time),label='GA')
	plt.plot(np.arange(start, end, step),np.array(bd_time),label='GD')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Training time')
	plt.title('Training time  vs. Iterations for Optimization Algorithms')
	plt.legend()
	plt.savefig('nn_trainTime_iterations.png')




def neural_network_rhc():

	breakpoint()




	rhc_train_acc1=[]
	rhc_test_acc1 = []

	for i in range( 1000, 50000, 5000):
		rhc_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='random_hill_climb', max_iters = 10000, bias = True, 
			is_classifier = True, learning_rate = 1e-03, early_stopping = True, 
			max_attempts = 1000, random_state =42, restarts =0)


		rhc_model.fit( X_train, y_train)

		y_predict = rhc_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		rhc_train_acc1.append( train_acc)

		#testing accuracy 
		y_predict = rhc_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		rhc_test_acc1.append( test_acc)


	print( "restart 0 done" )

	rhc_train_acc2=[]
	rhc_test_acc2 = []

	for i in range( 1000, 50000, 5000):
		rhc_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='random_hill_climb', max_iters = 10000, bias = True, 
			is_classifier = True, learning_rate = 1e-03, early_stopping = True, 
			max_attempts = 1000, random_state =42, restarts =2)


		rhc_model.fit( X_train, y_train)

		y_predict = rhc_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		rhc_train_acc2.append( train_acc)

		#testing accuracy 
		y_predict = rhc_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		rhc_test_acc2.append( test_acc)

	print( "restart 2 done" )

	rhc_train_acc3=[]
	rhc_test_acc3 = []

	for i in range( 1000, 50000, 5000):
		rhc_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='random_hill_climb', max_iters = 10000, bias = True, 
			is_classifier = True, learning_rate = 1e-03, early_stopping = True, 
			max_attempts = 1000, random_state =42, restarts =6)


		rhc_model.fit( X_train, y_train)

		y_predict = rhc_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		rhc_train_acc3.append( train_acc)

		#testing accuracy 
		y_predict = rhc_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		rhc_test_acc3.append( test_acc)

	print( "restart 6 done" )

	rhc_train_acc4=[]
	rhc_test_acc4 = []

	for i in range( 1000, 50000, 5000):
		rhc_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='random_hill_climb', max_iters = 10000, bias = True, 
			is_classifier = True, learning_rate = 1e-03, early_stopping = True, 
			max_attempts = 1000, random_state =42, restarts =8)


		rhc_model.fit( X_train, y_train)

		y_predict = rhc_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		rhc_train_acc4.append( train_acc)

		#testing accuracy 
		y_predict = rhc_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		rhc_test_acc4.append( test_acc)
	print( "restart 8 done" )

	rhc_train_acc5=[]
	rhc_test_acc5 = []

	for i in range( 1000, 50000, 5000):
		rhc_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='random_hill_climb', max_iters = 10000, bias = True, 
			is_classifier = True, learning_rate = 1e-03, early_stopping = True, 
			max_attempts = 1000, random_state =42, restarts =10)


		rhc_model.fit( X_train, y_train)

		y_predict = rhc_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		rhc_train_acc5.append( train_acc)

		#testing accuracy 
		y_predict = rhc_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		rhc_test_acc5.append( test_acc)

	print( "restart 10 done" )




	print( rhc_train_acc1)
	print(rhc_test_acc1)
	print( rhc_train_acc2)
	print(rhc_test_acc2)
	print( rhc_train_acc3)
	print(rhc_test_acc3)
	print( rhc_train_acc4)
	print(rhc_test_acc4)
	print( rhc_train_acc5)
	print(rhc_test_acc5)




	plt.figure()
	plt.grid()
	plt.plot(np.arange(1000,50000,5000),np.array(rhc_train_acc1),label='restart 0')
	plt.plot(np.arange(1000,50000,5000),np.array(rhc_train_acc2),label='restart 2')
	plt.plot(np.arange(1000,50000,5000),np.array(rhc_train_acc3),label='restart 4')
	plt.plot(np.arange(1000,50000,5000),np.array(rhc_train_acc4),label='restart 6')
	plt.plot(np.arange(1000,50000,5000),np.array(rhc_train_acc5),label='restart 8')
	plt.xlabel('iterations')
	plt.ylabel('Train Accuracy')
	plt.title('Train Accuracy vs. iterations for Random Hill Climbing')
	#plt.legend()
	plt.savefig('rhc_nn_restart_train_acc_with_itreations.png')
	plt.show()


	plt.figure()
	plt.grid()
	plt.plot(np.arange(1000,50000,5000),np.array(rhc_test_acc1),label='restart 0')
	plt.plot(np.arange(1000,50000,5000),np.array(rhc_test_acc2),label='restart 2')
	plt.plot(np.arange(1000,50000,5000),np.array(rhc_test_acc3),label='restart 4')
	plt.plot(np.arange(1000,50000,5000),np.array(rhc_test_acc4),label='restart 6')
	plt.plot(np.arange(1000,50000,5000),np.array(rhc_test_acc5),label='restart 8')
	plt.xlabel('iterations')
	plt.ylabel('Test Accuracy')
	plt.title('Test Accuracy vs. iterations for Random Hill Climbing')
	#plt.legend()
	plt.savefig('rhc_nn_restart_test_acc_with_itreations.png')
	plt.show()


def run_sa( r):
	sa_train_acc1=[]
	sa_test_acc1=[]

	time_sa1=[]

	print(" Exponential decay 0.6")

	schedule = mlrose.GeomDecay( 1000,r , 1)


	for i in range( 10, 500,10 ):

		sa_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='simulated_annealing', max_iters = i, bias = True, 
			is_classifier = True, learning_rate = 1e-03, 
			early_stopping = True, max_attempts = 1000, random_state = 42, schedule=schedule)

		st = time.time()

		sa_model.fit( X_train, y_train)
		et = time.time()
		time_exp = et-st
		time_sa1.append(time_exp )

		#predict labels 

		y_predict = sa_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		sa_train_acc1.append( train_acc)

		#testing accuracy 
		y_predict = sa_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		sa_test_acc1.append( test_acc)
	return np.array(sa_train_acc1), np.array(sa_test_acc1), np.array(time_sa1)




		

def neural_network_sa1():
	breakpoint()


	print(" Exponential decay 0.6")
	
	sa_train_acc2, sa_test_acc2, time2 = run_sa( 0.7)
	sa_train_acc1, sa_test_acc1, time1 = run_sa( 0.6)
	sa_train_acc3, sa_test_acc3, time3 = run_sa( 0.8)
	sa_train_acc4, sa_test_acc4, time4 = run_sa( 0.9)
	sa_train_acc5, sa_test_acc5, time5 = run_sa( 0.95)

	start = 10
	end = 500
	step =10





	



	print(sa_train_acc1)
	print( sa_test_acc1)
	print( time1/60)


	breakpoint()

	time1/=60
	time2/=60
	time3/=60
	time4/=60
	time5/=60


	



	plt.figure()
	plt.plot(np.arange(start,end,step),sa_train_acc1,label='r:0.6')
	plt.plot(np.arange(start,end,step),sa_train_acc2,label='r:0.7')
	plt.plot(np.arange(start,end,step),sa_train_acc3,label='r:0.8')
	plt.plot(np.arange(start,end,step),sa_train_acc4,label='r:0.9')
	plt.plot(np.arange(start,end,step),sa_train_acc5,label='r:0.95')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Train Accuracy')
	plt.title('Train Accuracy vs. Iterations for Different cooling exponent Algorithms')
	plt.legend()
	plt.savefig('nn_sa_train_cooling_exponent_iterations.png')
	plt.show()


	plt.figure()
	plt.plot(np.arange(start,end,step),sa_test_acc1,label='r:0.6')
	plt.plot(np.arange(start,end,step),sa_test_acc2,label='r:0.7')
	plt.plot(np.arange(start,end,step),sa_test_acc3,label='r:0.8')
	plt.plot(np.arange(start,end,step),sa_test_acc4,label='r:0.9')
	plt.plot(np.arange(start,end,step),sa_test_acc5,label='r:0.95')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Test Accuracy')
	plt.title('Test Accuracy vs. Iterations for Different cooling exponent Algorithms')
	plt.legend()
	plt.savefig('nn_sa_test_cooling_exponent_iterations.png')
	plt.show()


	plt.figure()
	plt.plot(np.arange(start,end,step),time1,label='r:0.6')
	plt.plot(np.arange(start,end,step),time2,label='r:0.7')
	plt.plot(np.arange(start,end,step),time3,label='r:0.8')
	plt.plot(np.arange(start,end,step),time4,label='r:0.9')
	plt.plot(np.arange(start,end,step),time5,label='r:0.95')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Time taken')
	plt.title('Time vs. Iterations for Different cooling exponent Algorithms')
	plt.legend()
	plt.savefig('nn_sa_time_cooling_exponent_iterations.png')
	plt.show()


	








def neural_network_sa():
	breakpoint()

	sa_train_acc_exp =[]
	sa_train_acc_geom=[]
	sa_train_acc_arith = []

	time_sa_arith=[]
	time_sa_geom=[]
	time_sa_exp=[]

	sa_test_acc_exp =[]
	sa_test_acc_geom=[]
	sa_test_acc_arith =[]

	print(" Exponential decay")


	for i in range( 1000, 30000, 5000):

		sa_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='simulated_annealing', max_iters = i, bias = True, 
			is_classifier = True, learning_rate = 1e-03, 
			early_stopping = True, max_attempts = 1000, random_state = 42, schedule=mlrose.ExpDecay())

		st = time.time()

		sa_model.fit( X_train, y_train)
		et = time.time()
		time_exp = et-st
		time_sa_exp.append(time_exp )

		#predict labels 

		y_predict = sa_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		sa_train_acc_exp.append( train_acc)

		#testing accuracy 
		y_predict = sa_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		sa_test_acc_exp.append( test_acc)

	breakpoint()


	print(" Geometric decay")




	for i in range( 1000, 30000, 5000):

		sa_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='simulated_annealing', max_iters = i, bias = True, 
			is_classifier = True, learning_rate = 1e-03, 
			early_stopping = True, max_attempts = 1000, random_state = 42, schedule=mlrose.GeomDecay())

		st=time.time()
		sa_model.fit( X_train, y_train)
		et = time.time()
		time_geom = et-st
		time_sa_geom.append(time_geom )

		#predict labels 

		y_predict = sa_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		sa_train_acc_geom.append( train_acc)

		#testing accuracy 
		y_predict = sa_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		sa_test_acc_geom.append( test_acc)


	print(" Arithmetic decay")



	for i in range( 1000, 30000, 5000):

		sa_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='simulated_annealing', max_iters = i, bias = True, 
			is_classifier = True, learning_rate = 1e-03, 
			early_stopping = True, max_attempts = 1000, random_state = 42, schedule=mlrose.ArithDecay())

		st=time.time()
		sa_model.fit( X_train, y_train)
		et = time.time()
		time_arith = et-st
		time_sa_arith.append(time_arith )

		#predict labels 

		y_predict = sa_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		sa_train_acc_arith.append( train_acc)

		#testing accuracy 
		y_predict = sa_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		sa_test_acc_arith.append( test_acc)



	print(sa_train_acc_arith)
	print( sa_test_acc_arith)

	print( sa_train_acc_geom)
	print( sa_test_acc_geom)

	print( sa_train_acc_exp)
	print( sa_test_acc_exp)

	time_sa_geom =np.array( time_sa_geom)
	time_sa_exp =np.array( time_sa_exp)
	time_sa_arith =np.array( time_sa_arith)


	time_sa_exp/=60
	time_sa_arith/=60
	time_sa_geom/=60



	plt.figure()
	plt.plot(np.arange(1000,30000,5000),np.array(sa_test_acc_exp),label='SA_EXP')
	plt.plot(np.arange(1000,30000,5000),np.array(sa_test_acc_geom),label='SA_GEOM')
	plt.plot(np.arange(1000,30000,5000),np.array(sa_test_acc_arith),label='SA_ARITH')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Test Accuracy')
	plt.title('Train Accuracy vs. Iterations for Optimization Algorithms')
	plt.legend()
	plt.savefig('nn_sa_schedule_iterations.png')
	plt.show()


	plt.figure()
	plt.plot(np.arange(1000,30000,5000),np.array(time_sa_exp),label='Time_EXP')
	plt.plot(np.arange(1000,30000,5000),np.array(time_sa_geom),label='Time_GEOM')
	plt.plot(np.arange(1000,30000,5000),np.array(time_sa_arith),label='Time_ARITH')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Time taken')
	plt.title('Time taken vs. Iterations for Different schedules')
	plt.legend()
	plt.savefig('nn_sa_schedule_time_taken_iterations.png')
	plt.show()


















def neural_network_ga():


	breakpoint()
	ga_train_acc1 =[]
	ga_test_acc1 =[]

	start = 10
	end = 1000
	step=10
	for i in range( start, end, step):
		ga_model = ga_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='genetic_alg', 
			max_iters = i, bias = True, is_classifier = True, learning_rate = 1e-03, early_stopping = True, 
			max_attempts = 1000, random_state = 42, pop_size=200, mutation_prob = 0.001)


		ga_model.fit( X_train, y_train)
		#predict labels 
		y_predict = ga_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		ga_train_acc1.append( train_acc)
		#testing accuracy 
		y_predict = ga_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		ga_test_acc1.append( test_acc)

	print("1 done")


	ga_train_acc2 =[]
	ga_test_acc2=[]
	for i in range( start, end, step):
		ga_model = ga_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='genetic_alg', 
			max_iters = i, bias = True, is_classifier = True, learning_rate = 1e-03, early_stopping = True, 
			max_attempts = 1000, random_state = 42, pop_size=500, mutation_prob = 0.001)


		ga_model.fit( X_train, y_train)
		#predict labels 
		y_predict = ga_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		ga_train_acc2.append( train_acc)
		#testing accuracy 
		y_predict = ga_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		ga_test_acc2.append( test_acc)

	print("2 done")

	ga_train_acc3 =[]
	ga_test_acc3 =[]
	for i in range( start, end, step):
		ga_model = ga_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='genetic_alg', 
			max_iters = i, bias = True, is_classifier = True, learning_rate = 1e-03, early_stopping = True, 
			max_attempts = 1000, random_state = 42, pop_size=200, mutation_prob = 0.01)


		ga_model.fit( X_train, y_train)
		#predict labels 
		y_predict = ga_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		ga_train_acc3.append( train_acc)
		#testing accuracy 
		y_predict = ga_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		ga_test_acc3.append( test_acc)

	print("3 done")

	ga_train_acc4 =[]
	ga_test_acc4 =[]
	for i in range( start, end, step):
		ga_model = ga_model = mlrose.NeuralNetwork( hidden_nodes = [2], activation='relu', algorithm ='genetic_alg', 
			max_iters = i, bias = True, is_classifier = True, learning_rate = 1e-03, early_stopping = True, 
			max_attempts = 1000, random_state = 42, pop_size=500, mutation_prob = 0.01)


		ga_model.fit( X_train, y_train)
		#predict labels 
		y_predict = ga_model.predict( X_train)
		train_acc = accuracy_score( y_train, y_predict)
		ga_train_acc4.append( train_acc)
		#testing accuracy 
		y_predict = ga_model.predict( X_test)
		test_acc = accuracy_score( y_test, y_predict)
		ga_test_acc4.append( test_acc)


	print("4 done")


	print( ga_train_acc1)
	print(ga_test_acc1)
	print( ga_train_acc2)
	print(ga_test_acc2)
	print( ga_train_acc3)
	print(ga_test_acc3)
	print( ga_train_acc4)
	print(ga_test_acc4)



	plt.figure()
	plt.grid()
	plt.plot(np.arange(start, end, step),np.array(ga_train_acc1),label='pop :200/mutation_prob 0.001')
	plt.plot(np.arange(start, end, step),np.array(ga_train_acc2),label='pop :500/mutation_prob 0.001')
	plt.plot(np.arange(start, end, step),np.array(ga_train_acc3),label='pop :200/mutation_prob 0.01')
	plt.plot(np.arange(start, end, step),np.array(ga_train_acc4),label='pop :500/mutation_prob 0.01')
	plt.xlabel('iterations')
	plt.ylabel('Train Accuracy')
	plt.title('Train Accuracy vs. iterations for genetic_alg')
	#plt.legend()
	plt.savefig('ga_tuning_nn_pop_mutation_train_acc_with_itreations.png')
	plt.show()


	plt.figure()
	plt.grid()
	plt.plot(np.arange(start, end, step),np.array(ga_test_acc1),label='pop :200/mutation_prob 0.001')
	plt.plot(np.arange(start, end, step),np.array(ga_test_acc2),label='pop :500/mutation_prob 0.001')
	plt.plot(np.arange(start, end, step),np.array(ga_test_acc3),label='pop :200/mutation_prob 0.01')
	plt.plot(np.arange(start, end, step),np.array(ga_test_acc4),label='pop :500/mutation_prob 0.01')
	plt.xlabel('iterations')
	plt.ylabel('Test Accuracy')
	plt.title('Test Accuracy vs. iterations for genetic_alg')
	#plt.legend()
	plt.savefig('ga_tuning_nn_pop_mutation_test_acc_with_itreations.png')
	plt.show()






















warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


#neural_network_rhc()
#neural_network_sa1()

neural_network_iterations()
#neural_network_ga()










	





































