import numpy as np 
import mlrose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, datasets
import time
from random import randint
import warnings





def our_fitness_func(state):
    global eval_count
    fitness = ml.FourPeaks(t_pct=0.15)
    eval_count += 1
    return fitness.evaluate(state)




#finding the optimal parameters  for rhc 
def find_optimal_parameters_rhc(problem, n,name):
	init_state = np.random.randint(2,size=n)
	print("RHC started")

	attempts=1000
	iters = 10000

	fitness_curve_arr = []
	fitness_value =[]

	for i in range( 0,25, 5):
		best_state, best_fitness, fitness_curve= mlrose.random_hill_climb(problem, restarts =i,max_attempts =attempts, max_iters=iters, init_state = init_state, curve=True)
		fitness_curve_arr.append(fitness_curve)

		fitness_value.append( best_fitness)

	print( fitness_value)


	fitness_value=np.array( fitness_value)
	print( fitness_curve_arr)
	plt.figure()
	plt.grid()
	plt.plot(fitness_curve_arr[0], label ='restarts 0')
	plt.plot(fitness_curve_arr[1], label ='restarts 5')
	plt.plot(fitness_curve_arr[2], label ='restarts 10')
	plt.plot(fitness_curve_arr[3], label ='restarts 15')
	plt.plot(fitness_curve_arr[4],label ='restarts 20')
	plt.xlabel( 'iterations')
	plt.ylabel('fitness value ')
	plt.legend()
	plt.title('variation of fitness with random restarts')
	plt.show()
	plt.savefig(' optimal_rhc'+ name+'.png')
	print("RHC done")


def four_peaks_compare_algorithms(problem ,ga_param, sa_param, rhc_param, mimic_param, name):
	fitness_sa_arr = []
	fitness_rhc_arr = []
	fitness_ga_arr = []
	fitness_mimic_arr = []
	attempts =1000
	iters =20000

	time_sa_arr = []
	time_rhc_arr = []
	time_ga_arr = []
	time_mimic_arr = []
	for n in range(5,120,20):
		fitness = mlrose.FourPeaks(t_pct=0.15)
		print(n,"started")
		problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize=True, max_val=2)
		init_state = np.random.randint(2,size=n)
		schedule = mlrose.GeomDecay( 1000, sa_param[0], 1)
		st = time.time()
		best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = attempts, max_iters=iters, init_state = init_state, curve=True)
		end = time.time()
		sa_time = end-st

		st = time.time()
		best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts = attempts,restarts=rhc_param[0], max_iters=iters, init_state = init_state, curve=True)
		end = time.time()
		rhc_time = end-st

		st = time.time()
		best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts = attempts, 
			max_iters=iters, curve=True, pop_size=ga_param[0], mutation_prob=ga_param[1])
		end = time.time()
		ga_time = end-st

		st = time.time()
		best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,pop_size=mimic_param[0], max_attempts = attempts, 
			max_iters=iters,keep_pct=mimic_param[1], curve=True, fast_mimic=True)
		end = time.time()
		mimic_time = end-st
		print(mimic_time,n)
		print(n,"done")

		fitness_sa_arr.append(best_fitness_sa)
		fitness_rhc_arr.append(best_fitness_rhc)
		fitness_ga_arr.append(best_fitness_ga)
		fitness_mimic_arr.append(best_fitness_mimic)

		time_sa_arr.append(sa_time)
		time_rhc_arr.append(rhc_time)
		time_ga_arr.append(ga_time)
		time_mimic_arr.append(mimic_time)

	fitness_sa_arr = np.array(fitness_sa_arr)
	fitness_rhc_arr = np.array(fitness_rhc_arr)
	fitness_ga_arr = np.array(fitness_ga_arr)
	fitness_mimic_arr = np.array(fitness_mimic_arr)

	time_sa_arr = np.array(time_sa_arr)
	time_rhc_arr = np.array(time_rhc_arr)
	time_ga_arr = np.array(time_ga_arr)
	time_mimic_arr = np.array(time_mimic_arr)

	plt.figure()
	plt.plot(np.arange(5,120,20),fitness_sa_arr,label='SA')
	plt.plot(np.arange(5,120,20),fitness_rhc_arr,label = 'RHC')
	plt.plot(np.arange(5,120,20),fitness_ga_arr, label = 'GA')
	plt.plot(np.arange(5,120,20),fitness_mimic_arr, label = 'MIMIC')
	plt.xlabel('Input Size')
	plt.ylabel('Fitness Vaue')
	plt.legend()
	plt.title('Fitness Value vs. Input Size (Conti Peaks)')
	plt.savefig('ContinuousPeaks_input_size_fitness.png')
	plt.show()

	plt.figure()
	plt.plot(np.arange(5,120,20),time_sa_arr,label='SA')
	plt.plot(np.arange(5,120,20),time_rhc_arr,label='RHC')
	plt.plot(np.arange(5,120,20),time_ga_arr,label='GA')
	plt.plot(np.arange(5,120,20),time_mimic_arr,label='MIMIC')
	plt.legend()
	plt.xlabel('Input Size')
	plt.ylabel('Computation Time (s)')
	plt.title('Computation Time vs. Input Size (Conti Peaks)')
	plt.savefig('continuousPeaks_input_size_computation.png')
	plt.show()












def find_optimal_parameters_ga_pop(problem, name):
    print("GA Started")
    population_size = [200, 500]
    attempts = 1000
    iters = 10000
    fitness_value=[]
    fitness_curve_arr =[]
    for p in population_size:
        best_state, best_fitness_ga, fitness_curve= mlrose.genetic_alg(problem, pop_size =p, mutation_prob = 0.001,
        	max_attempts =attempts, max_iters=iters, curve=True)
        fitness_value.append( best_fitness_ga)
        fitness_curve_arr.append( fitness_curve)


    for p in population_size:
        best_state, best_fitness_ga, fitness_curve= mlrose.genetic_alg(problem, pop_size =p, mutation_prob = 0.01,
        	max_attempts =attempts, max_iters=iters, curve=True)
        fitness_value.append( best_fitness_ga)
        fitness_curve_arr.append( fitness_curve)
    
    print( fitness_value)

    print( fitness_curve_arr)
    
    fitness_value = np.array(fitness_value)
    breakpoint()



    plt.figure()
    plt.grid()
    plt.plot(fitness_curve_arr[0], label =' pop 200: mutation_prob:0.001')
    plt.plot(fitness_curve_arr[1], label=' pop 500: mutation_prob:0.001')
    plt.plot(fitness_curve_arr[2], label=' pop 200: mutation_prob:0.01')
    plt.plot(fitness_curve_arr[3], label=' pop 500: mutation_prob:0.01')
    plt.legend()
    plt.xlabel( 'iterations')
    plt.ylabel('fitness value ')
    plt.title('variation of fitness with Mutation and population size')
    plt.show()
    plt.savefig(' optimal_ga'+ name+'.png')
    print("GA Done")
    




def sa_different_schedule(problem, name, n):

	fitness_curve_arr =[]
	fitness_values =[]

	schedule =[ mlrose.GeomDecay(),mlrose.ArithDecay(),mlrose.ExpDecay()]
	init_state = np.random.randint( 2, size=n)
	for s in schedule:
		best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = s, 
			max_attempts = 1000, max_iters=10000, init_state = init_state, curve=True)
		fitness_curve_arr.append( fitness_curve)
		fitness_values.append( best_fitness)

	print( fitness_values)



	plt.figure()
	plt.grid()
	plt.plot(fitness_curve_arr[0],label='Geom')
	plt.plot(fitness_curve_arr[1],label = 'Arith')
	plt.plot(fitness_curve_arr[2], label = 'EXP')
	plt.xlabel('iterations')
	plt.ylabel('fitness values')
	plt.legend()
	plt.title('Fitness values  vs. Different schedule')
	plt.savefig(name+'sa_optimum_diff_schedules.png')
	plt.show()

def find_optimal_parameters_sa(problem, n, name):
    print("SA  Started")
    init_state = np.random.randint( 2, size=n)
    decay = [0.65,0.7,0.8, 0.9, 0.95]
    fitness_value=[]
    fitness_curve_arr = []
    for r in decay:
        schedule = mlrose.GeomDecay( 10000, r, 1)
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing( problem,schedule=schedule, max_attempts=2000, max_iters=100000,init_state=init_state, curve=True) 
        fitness_value.append( best_fitness)
        fitness_curve_arr.append(fitness_curve)
    fitness_value=np.array( fitness_value)
    print( fitness_value)

    plt.figure()
    plt.grid()
    plt.plot(fitness_curve_arr[0], label ='r:0.65')
    plt.plot(fitness_curve_arr[1], label='r:0.7')
    plt.plot(fitness_curve_arr[2], label ='r:0.8')
    plt.plot(fitness_curve_arr[3], label ='r:0.9')
    plt.plot(fitness_curve_arr[4], label ='r:0.95')
    plt.legend()
    plt.xlabel( 'iterations')
    plt.ylabel('fitness value ')
    plt.title('variation of fitness with various colling exponents')
    plt.show()
    plt.savefig(' optimal_sa'+ name+'.png')
    print("SA done")




def find_optimal_parameters_mimic( problem,n, name):
    print("Mimic Started")
    population_size =[200, 500]
    fitness_values = []
    fitness_curve_arr =[]
    
    for p in population_size:
        best_state, best_fitness= mlrose.mimic( problem, pop_size=p, keep_pct=0.1, max_attempts=1000, max_iters=10000, fast_mimic=True)
        fitness_values.append( best_fitness)
        fitness_curve_arr.append(fitness_curve)

    for p in population_size:
        best_state, best_fitness= mlrose.mimic( problem, pop_size=p, keep_pct=0.2, max_attempts=1000, max_iters=10000, fast_mimic=True)
        fitness_values.append( best_fitness)
        fitness_curve_arr.append(fitness_curve)

    for p in population_size:
        best_state, best_fitness= mlrose.mimic( problem, pop_size=p, keep_pct=0.5, max_attempts=1000, max_iters=10000, fast_mimic=True)
        fitness_values.append( best_fitness)
        fitness_curve_arr.append(fitness_curve)
    
    fitness_values=np.array( fitness_values)
    print(fitness_values)

    plt.figure()
    plt.grid()
    plt.plot(fitness_curve_arr[0], label =' pop 200: keep pct 0.1')
    plt.plot(fitness_curve_arr[1], label=' pop 500: keep pct 0.1')
    plt.plot(fitness_curve_arr[2], label=' pop 200: keep pct 0.2')
    plt.plot(fitness_curve_arr[3], label=' pop 500: keep pct 0.2')
    plt.plot(fitness_curve_arr[4], label=' pop 200: keep pct 0.5')
    plt.plot(fitness_curve_arr[5], label=' pop 500: keep pct 0.5')
    plt.legend()
    
    plt.xlabel( 'iterations')
    plt.ylabel('fitness value ')
    plt.title('variation of fitness with Mutation and keep pct values ')
    plt.show()
    
    plt.savefig(' optimal_mimic_'+ name+'.png')
    print("Mimic Done")




def compare_algorithms_iterations( problem, ga_param, sa_param, rhc_param, mimic_param, name, n):

	attempts=1000
	iters =1000
	schedule = mlrose.GeomDecay( 1000 ,sa_param[0], 1)

	init_state = init_state = np.random.randint( 2, size=n)

	st = time.time()

	print(" Started")

	best_state, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, pop_size =ga_param[0], mutation_prob = ga_param[1],max_attempts =attempts, max_iters=iters, curve=True)
	et =time.time()
	ga_time = et-st

	print("Genetic done")
	st = time.time()


	best_state, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule,init_state=init_state,max_attempts =attempts, max_iters=iters, curve=True)
	et= time.time()
	sa_time = et-st

	print(" SA done")

	st = time.time()


	best_state, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, restarts =rhc_param[0], init_state= init_state,max_attempts =attempts, max_iters=iters, curve=True)
	et = time.time()
	rhc_time = et-st

	print(" RHC done")
	st = time.time()





	best_state, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem, pop_size =mimic_param[0], keep_pct = mimic_param[1],max_attempts =attempts, max_iters=iters, curve=True, fast_mimic=True)
	et = time.time()
	mimic_time = et-st


	print(" ALL done ")


	

	print( ga_time, sa_time, rhc_time,mimic_time) 


	plt.figure()
	plt.plot(fitness_curve_sa,label='SA')
	plt.plot(fitness_curve_rhc,label = 'RHC')
	plt.plot(fitness_curve_ga, label = 'GA')
	plt.plot(fitness_curve_mimic, label = 'MIMIC')
	plt.xlabel('iterations ')
	plt.ylabel('fitness values ')
	plt.legend()
	plt.title('fitness values vs. iterations'+ name)
	#plt.savefig(name+'fitness_VS_iterations.png')
	plt.show()



def compare_algorithms_tpct( ga_param, sa_param, rhc_param, mimic_param, name):


	T_value = [0.1, 0.2, 0.3, 0.4, 0.5]
	init_state = np.random.randint( 2, size=100)
	attempts =1000
	iters =1000

	fitness_ga =[]
	fitness_sa =[]
	fitness_rhc = []
	fitness_mimic = []
	schedule = mlrose.GeomDecay( 1000, sa_param[0], 1)


	for t in T_value:
		fitness = mlrose.FourPeaks( t_pct =t)
		print(t)
		problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)


		best_state, best_fitness_ga = mlrose.genetic_alg( problem, pop_size = ga_param[0], mutation_prob=ga_param[1],
			max_attempts=attempts, max_iters=iters )
		fitness_ga.append( best_fitness_ga)
		print('ga done')

		best_state, best_fitness_sa = mlrose.simulated_annealing( problem , schedule = schedule, init_state=init_state,
			max_attempts=attempts, max_iters=iters)
		fitness_sa.append( best_fitness_sa)
		print('sa done')

		best_state, best_fitness_rhc = mlrose.random_hill_climb( problem, init_state=init_state, restarts=rhc_param[0],
			max_attempts=attempts, max_iters=iters)
		fitness_rhc.append( best_fitness_rhc)
		print('rhc done')
		best_state, best_fitness_mimic = mlrose.mimic( problem, pop_size=mimic_param[0], keep_pct=mimic_param[1], 
			max_attempts=attempts, max_iters=iters, fast_mimic=True)
		fitness_mimic.append( best_fitness_mimic)
		print('loop completed')


	plt.figure()
	plt.xlabel(' t_pct values ')
	plt.ylabel(' best fitness value ')
	plt.plot( T_value, fitness_ga, label='GA')
	plt.plot( T_value, fitness_sa, label='SA')
	plt.plot( T_value, fitness_rhc, label='RHC')
	plt.plot( T_value, fitness_mimic, label='MIMIC')
	plt.legend()
	plt.title( ' t_pct values variation with fitness')
	plt.savefig('4peaks_tpct_fitnesss.png')
	plt.show()



def compare_algorithms_func_eval( problem, ga_param, sa_param, rhc_param, mimic_param, name):
	# comparing function on function evalutaions 


	# they all contains the best params for each algorithm 

	func_eval_ga= []
	init_state = np.random.randint( 2, size=n)
	func_eval_sa =[]

	func_eval_mimic = []
	func_eval_rhc= []
	schedule = mlrose.GeomDecay( 1000 ,sa_param[0], 1)


	for n in range( 40 , 101, 10):
		fitness = mlrose.CustomFitness(our_fitness_func)
		problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True)
		
		eval_count = 0
		best_state, best_fitness= mlrose.genetic_alg( problem, pop_size=ga_param[0], mutation_prob=ga_param[1], max_attempts=1000,max_iters =100000 )
		func_eval_ga.append( eval_count)

		eval_count = 0
		best_state, best_fitness= mlrose.simulated_annealing( problem, schedule=schedule, max_attempts=1000 ,max_iters =100000 , init_state=init_state)
		func_eval_sa.append( eval_count)

		eval_count = 0
		best_state, best_fitness= mlrose.mimic( problem, pop_size=mimic_param[0], keep_pct=mimic_param[1], max_attempts=1000
		,max_iters =100000 )
		func_eval_mimic.append( eval_count)

		eval_count = 0
		best_state, best_fitness = mlrose.random_hill_climb( problem, restarts=rhc_param[0], init_state=init_state, max_attempts=1000
		,max_iters =100000 )
		func_eval_rhc.append( eval_count)


	plt.figure()
	plt.plot(np.arange(40,101,10),func_eval_sa,label='SA')
	plt.plot(np.arange(40,101,10),func_eval_rhc,label = 'RHC')
	plt.plot(np.arange(40,101,10),func_eval_ga, label = 'GA')
	plt.plot(np.arange(40,101,10),func_eval_mimic, label = 'MIMIC')
	plt.xlabel('problem size ')
	plt.ylabel('function evaluations')
	plt.legend()
	plt.title('Function evalutaions vs. Input Size (4 Peaks)')
	plt.savefig(name+'func_eval_VS_input_size_fitness.png')
	plt.show()












def continuous_peaks():
    breakpoint()
    n =100
    fitness = mlrose.ContinuousPeaks( t_pct =0.15)
    problem = mlrose.DiscreteOpt(length = n, fitness_fn =fitness, maximize = True, max_val =2)
    ga_param = [500, 0.1]
    sa_param = [0.85]
    mimic_param= [500, 0.2]
    rhc_param = [15]
    breakpoint()

    #compare_algorithms_iterations(problem, ga_param, sa_param, rhc_param, mimic_param,'continuouspeaks', n)
    #compare_algorithms_tpct( ga_param, sa_param, rhc_param, mimic_param, 'continuousPeaks')
    #four_peaks_compare_algorithms(problem ,ga_param, sa_param, rhc_param, mimic_param, 'continuousPeaks')

    #sa_different_schedule( problem, 'continuousPeaks', 100)
    #find_optimal_parameters_ga_pop(problem, 'continuousPeaks')
    #find_optimal_parameters_rhc( problem, 100, 'continuousPeaks')
    #find_optimal_parameters_sa( problem, 100, 'continuousPeaks')
    #find_optimal_parameters_mimic( problem, 100, 'continuousPeaks')
    
    














continuous_peaks()


