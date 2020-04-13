
import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns


from gym.envs.toy_text.frozen_lake import generate_random_map

from mdp import PolicyIteration

import mdptoolbox, mdptoolbox.example

from mdp import ValueIteration






def Forest_Experiments():
  
  print('POLICY ITERATION WITH FOREST MANAGEMENT')
  P, R = mdptoolbox.example.forest(S = 2000)
  value_f = []
  policy = []
  iters = []
  time_array = []

  threshold_arr = [0.1,0.01,0.001,0.0001,0.00001]
  for i in [0.00000001,0.01,0.001,0.0001,0.00001]:
    pi = ValueIteration(P, R, 0.9,epsilon =i)
    pi.run()
    value_f.append(np.mean(pi.V))
    policy.append(pi.policy)
    iters.append(pi.iter)
    time_array.append(pi.time)


  plt.plot(threshold_arr, time_array)
  plt.xlabel('threshold')
  plt.title('Forest Management - Value Iteration - Execution Time Analysis')
  plt.ylabel('Execution Time (s)')
  plt.grid()
  plt.show()

  
  plt.plot(threshold_arr,value_f)
  plt.xlabel('threshold')
  plt.ylabel('Average Rewards')
  plt.title('Forest Management - Value Iteration - Reward Analysis')
  plt.grid()
  plt.show()

  plt.plot(threshold_arr,iters)
  plt.xlabel('threshold')
  plt.ylabel('Iterations to Converge')
  plt.title('Forest Management - Value Iteration - Convergence Analysis')
  plt.grid()
  plt.show()

  print('VALUE ITERATION WITH FOREST MANAGEMENT')
  P, R = mdptoolbox.example.forest(S=2000)
  value_f = [0]*10
  policy = [0]*10
  iters = [0]*10
  time_array = [0]*10
  gamma_arr = [0] * 10
  for i in range(0,10):
    pi = ValueIteration(P, R, 0.95, epsilon = 0.00001)
    pi.run()
    gamma_arr[i]=(i+0.5)/10
    value_f[i] = np.mean(pi.V)
    policy[i] = pi.policy
    iters[i] = pi.iter
    time_array[i] = pi.time


  plt.plot(gamma_arr, time_array)
  plt.xlabel('Gammas')
  plt.title('Forest Management - Value Iteration - Execution Time Analysis')
  plt.ylabel('Execution Time (s)')
  plt.grid()
  plt.show()
  
  plt.plot(gamma_arr,value_f)
  plt.xlabel('Gammas')
  plt.ylabel('Average Rewards')
  plt.title('Forest Management - Value Iteration - Reward Analysis')
  plt.grid()
  plt.show()

  plt.plot(gamma_arr,iters)
  plt.xlabel('Gammas')
  plt.ylabel('Iterations to Converge')
  plt.title('Forest Management - Value Iteration - Convergence Analysis')
  plt.grid()
  plt.show()



breakpoint()
Forest_Experiments()