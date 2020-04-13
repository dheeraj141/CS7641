
import time , sys, os
import hiive.mdptoolbox, hiive.mdptoolbox.example


import numpy as np

import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from gym.envs.toy_text.frozen_lake import generate_random_map


#from hiive.mdptoolbox.mdp import QLearning
#import mdptoolbox, mdptoolbox.example
from mdp import QLearning




def plot_graph( x, y, title, x_label, y_label):

    plt.plot(x, y)

    plt.xlabel(x_label)

    plt.title(title)

    plt.ylabel(y_label)


    plt.grid()

    plt.show()

def Q_learning_forest_management():
    print('Q LEARNING WITH FOREST MANAGEMENT')


    #P, R = hiive.mdptoolbox.example.forest(3, 4, 2,0.8)


    #ql = QLearning(P, R, 0.96)

    #ql.run()
    #ql.reward_array 
    #ql.policy
    P, R = hiive.mdptoolbox.example.forest(S=2000,p=0.1)
    value_f = []
    policy = []
    iters = []
    time_array = []
    Q_table = []
    rew_array = []
    epsilon_array = [0.5]

    epsilon = 0.7
    for gamma in [0.95]:

        alpha_arr = [0.9,0.5,0.7,0.8,0.9, 0.95]


        for alpha in alpha_arr:

            st = time.time()

            pi = QLearning(P,R,gamma = gamma, epsilon =epsilon, n_iter = 30000, alpha =alpha)


            end = time.time()

            pi.run()

            value_f.append(np.mean(pi.V))

            policy.append(pi.policy)

            time_array.append(end-st)

            Q_table.append(pi.Q)


        plot_graph( alpha_arr, time_array, " Execution time with alpha forest management", "alpha", "time (s)")


        plot_graph( alpha_arr, value_f, " V values with alpha", " alpha ", "v values ")




    plt.subplot(1,6,1)
    plt.imshow(Q_table[0][:20,:])
    plt.title('Epsilon=0.05')

    plt.subplot(1,6,2)
    plt.title('Epsilon=0.15')
    plt.imshow(Q_table[1][:20,:])

    plt.subplot(1,6,3)
    plt.title('Epsilon=0.25')
    plt.imshow(Q_table[2][:20,:])

    plt.subplot(1,6,4)
    plt.title('Epsilon=0.50')
    plt.imshow(Q_table[3][:20,:])

    plt.subplot(1,6,5)
    plt.title('Epsilon=0.75')
    plt.imshow(Q_table[4][:20,:])

    plt.subplot(1,6,6)
    plt.title('Epsilon=0.95')
    plt.imshow(Q_table[5][:20,:])
    plt.colorbar()
    plt.show()




breakpoint()
Q_learning_forest_management()