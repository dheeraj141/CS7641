import numpy as np 
import mlrose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, datasets
import time
from random import randint
import warnings





#finding the optimal parameters  for rhc 
def find_optimal_parameters_rhc(problem, n,name):
    
    print("RHC started")

    attempts=5000
    iters = 50000

    fitness_curve_arr = []
    fitness_value =[]

    for i in range( 0,25, 5):
                init_state = np.random.randint(4,size=n)
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
    plt.legend()
    plt.xlabel( 'iterations')
    plt.ylabel('fitness value ')
    plt.title('variation of fitness with random restarts')
    plt.show()
    plt.savefig(' optimal_rhc'+ name+'.png')
    print("RHC done")







def find_optimal_parameters_ga_pop(problem, name):
    print("GA Started")
    population_size = [200, 500]
    attempts = 1000
    iters = 1000
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
    init_state = np.random.randint(4, size=n)
    for s in schedule:
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = s, 
            max_attempts = 1000, max_iters=50000, init_state = init_state, curve=True)
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
    init_state = np.random.randint( 4, size=n)
    decay = [0.65,0.7,0.8, 0.9, 0.95]
    fitness_value=[]
    fitness_curve_arr = []
    for r in decay:
        schedule = mlrose.GeomDecay( 1000, r, 1)
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing( problem,schedule=schedule, max_attempts=1000, max_iters=1000,init_state=init_state, curve=True) 
        fitness_value.append( best_fitness)
        fitness_curve_arr.append(fitness_curve)
    fitness_value=np.array( fitness_value)
    print( fitness_value)

    plt.figure()
    plt.grid()
    plt.plot(fitness_curve_arr[0], label ='r:0.65')
    plt.plot(fitness_curve_arr[1], label='r:0.7')
    plt.plot(fitness_curve_arr[2], label='r:0.8')
    plt.plot(fitness_curve_arr[3], label='r:0.9')
    plt.plot(fitness_curve_arr[4], label='r:0.95')
    plt.legend()
    plt.xlabel( 'iterations')
    plt.ylabel('fitness value ')
    plt.title('variation of fitness with various cooling exponent')
    plt.show()
    plt.savefig(' optimal_sa_cooling_exponent'+ name+'.png')
    print("SA done")




def find_optimal_parameters_mimic( problem,n, name):
    print("Mimic Started")
    population_size =[200, 500]
    fitness_values = []
    fitness_curve_arr =[]
    
    for p in population_size:
        best_state, best_fitness, fitness_curve = mlrose.mimic( problem, pop_size=p, keep_pct=0.1, max_attempts=100, max_iters=100, curve=True)
        fitness_values.append( best_fitness)
        fitness_curve_arr.append(fitness_curve)

    for p in population_size:
        best_state, best_fitness, fitness_curve = mlrose.mimic( problem, pop_size=p, keep_pct=0.2, max_attempts=100, max_iters=100, curve=True)
        fitness_values.append( best_fitness)
        fitness_curve_arr.append(fitness_curve)

    for p in population_size:
        best_state, best_fitness, fitness_curve = mlrose.mimic( problem, pop_size=p, keep_pct=0.5, max_attempts=100, max_iters=100, curve=True)
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





# compare algorithm on convergence 
def compare_algorithms_iterations( problem, ga_param, sa_param, rhc_param, mimic_param, name, n):

    attempts=500
    iters =500
    schedule = mlrose.GeomDecay( 1000 ,sa_param[0], 1)

    init_state = init_state = np.random.randint( 4, size=n)

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
    plt.grid()
    plt.plot(fitness_curve_sa,label='SA')
    plt.plot(fitness_curve_rhc,label = 'RHC')
    plt.plot(fitness_curve_ga, label = 'GA')
    plt.plot(fitness_curve_mimic, label = 'MIMIC')
    plt.xlabel('iterations ')
    plt.ylabel('fitness values ')
    plt.legend()
    plt.title('fitness values vs. iterations'+ name)
    plt.savefig(name+'fitness_VS_iterations.png')
    plt.show()



# compare the algorithms on problem size 
def four_peaks_compare_algorithms(ga_param, sa_param, rhc_param, mimic_param, name):
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
    start_n = 40
    end_n = 121
    step_n = 20
    for n in range(start_n,end_n,step_n):
                weights = np.random.randint( 1, 50, size=n)
                values = np.random.randint( 1,50, size=n)
                max_weight = 0.6
                fitness = mlrose.Knapsack( weights, values, max_weight)
                problem = mlrose.DiscreteOpt(length = n, fitness_fn =fitness, maximize = True, max_val =2)
                print(n,"started")
                init_state = np.random.randint(2,size=n)
                schedule = mlrose.GeomDecay( 1000, sa_param[0], 1)
                st = time.time()
                best_state_sa, best_fitness_sa = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = attempts, max_iters=iters, init_state = init_state)
                end = time.time()
                sa_time = end-st

                st = time.time()
                best_state_rhc, best_fitness_rhc = mlrose.random_hill_climb(problem, max_attempts = attempts,restarts=rhc_param[0], max_iters=iters, init_state = init_state)
                end = time.time()
                rhc_time = end-st

                st = time.time()
                best_state_ga, best_fitness_ga = mlrose.genetic_alg(problem, max_attempts = attempts, max_iters=iters, pop_size=ga_param[0], mutation_prob=ga_param[1])
                end = time.time()
                ga_time = end-st

                st = time.time()
                best_state_mimic, best_fitness_mimic= mlrose.mimic(problem,pop_size=mimic_param[0], max_attempts = attempts, max_iters=iters,keep_pct=mimic_param[1],  fast_mimic=True)
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


    breakpoint()


    plt.figure()
    plt.plot(np.arange(start_n,end_n,step_n),fitness_sa_arr,label='SA')
    plt.plot(np.arange(start_n,end_n,step_n),fitness_rhc_arr,label = 'RHC')
    plt.plot(np.arange(start_n,end_n,step_n),fitness_ga_arr, label = 'GA')
    plt.plot(np.arange(start_n,end_n,step_n),fitness_mimic_arr, label = 'MIMIC')
    plt.xlabel('Input Size')
    plt.ylabel('Fitness Vaue')
    plt.legend()
    plt.title('Fitness Value vs. Input Size (Knapsack)')
    plt.savefig('Knapsack_input_size_fitness.png')
    plt.show()

    plt.figure()
    plt.plot(np.arange(start_n,end_n,step_n),time_sa_arr,label='SA')
    plt.plot(np.arange(start_n,end_n,step_n),time_rhc_arr,label='RHC')
    plt.plot(np.arange(start_n,end_n,step_n),time_ga_arr,label='GA')
    plt.plot(np.arange(start_n,end_n,step_n),time_mimic_arr,label='MIMIC')
    plt.legend()
    plt.xlabel('Input Size')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time vs. Input Size (Knapsack)')
    plt.savefig('KNapsack_input_size_computation.png')
    plt.show()













def knapsack():
    n =50
    breakpoint()

    # particular example used for comparing algorithms

    #weights =[ 80, 82, 85, 70, 72, 70, 66, 50, 55, 25, 50, 55, 40, 48, 50, 32, 22, 60, 30, 32, 40, 38, 35, 32, 25, 28, 30, 22,50, 30, 45, 30, 60, 50, 20, 65, 20, 25, 30, 10, 20, 25, 15, 10, 10, 10, 4, 4, 2, 1]

    #values=[220, 208, 198, 192, 180,180, 165, 162, 160, 158, 155, 130, 125, 122, 120, 118,115, 110, 105, 101, 100, 100, 98, 96, 95, 90, 88, 82, 80, 77,75, 73, 72, 70, 69, 66, 65, 63, 60, 58, 56, 50, 30, 20, 15, 10, 8, 5, 3, 1]
    #W = 1000

    # one more
    #weights =[23,47,14,34,46,46,6,45,4,5,13,47,30,22,38,46,46,3,3,24,37,45,8,17,39,40,28,34,48,23,44,21,33,19,24,28,32,44,3,12,12,13,4,9,36,7,20,4,24,33]



    #values = [2,15,5,1,2,42,8,18,48,30,42,8,38,41,12,19,47,48,37,26,29,22,40,49,16,47,3,42,46,15,18,45,13,13,47,31,21,41,35,25,33,3,20,30,29,22,44,40,31,46]

    #sum = np.sum( weights)
    #print( sum)

    # optimal parameters found for the problem
    ga_param = [500,0.001]
    sa_param = [0.9]
    rhc_param =[15]
    mimic_param =[500,0.2]
    #max_weight = W/sum
    #print( sum*max_weight)
    #fitness = mlrose.Knapsack( weights, values, max_weight)
    #problem = mlrose.DiscreteOpt(length = n, fitness_fn =fitness, maximize = True, max_val =2)


    # all the functions call 
    #compare_algorithms_iterations( problem, ga_param, sa_param, rhc_param, mimic_param, 'knapsack', n)
    #four_peaks_compare_algorithms(ga_param, sa_param, rhc_param, mimic_param, 'knapsack')
    
    #sa_different_schedule(problem,'knapsack', n) 
    #find_optimal_parameters_ga_pop(problem, 'knapsack')
    #find_optimal_parameters_rhc( problem, n, 'knapsack')
    #find_optimal_parameters_sa( problem, n, 'knapsack')
    #find_optimal_parameters_mimic( problem, n, 'knapsack')



knapsack()
    


