




import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from gym.envs.toy_text.frozen_lake import generate_random_map



def plot_graph( x, y, title, x_label, y_label):
  plt.plot(x, y)
  plt.xlabel(x_label)
  plt.title(title)
  plt.ylabel(y_label)
  plt.grid()
  plt.show()




def run_episodes(env, V, policy, num_games=100):
    tot_rew = 0
    state = env.reset()
    for _ in range(num_games):
        done = False
        while not done:
            next_state, reward, done, _ = env.step(policy[state])
            state = next_state
            tot_rew += reward 
            if done:
                state = env.reset()
    print('Won %i of %i games!'%(tot_rew, num_games))

    return tot_rew




def Q_learning_frozen_lake(env):
  ### Q-LEARNING #####
  print('Q LEARNING WITH FROZEN LAKE')
  st = time.time()
  reward_array = []
  iter_array = []
  size_array = []
  chunks_array = []
  averages_array = []
  time_array = []
  Q_array = []
  y = [0.05,0.15,0.25,0.5,0.75,0.90]


  games = []

  gamma = 0.95



  
  for epsilon in [0.2]:

    alpha_arr = [0.8,0.4,0.6,0.8,0.95]


    for alpha in alpha_arr:

      error = []

      st = time.time()

      Q = np.zeros((env.observation_space.n, env.action_space.n))

      rewards = []

      iters = []

      optimal=[0]*env.observation_space.n
      episodes = 30000
      environment  = 'FrozenLake-v0'
      env = gym.make(environment)
      env = env.unwrapped
      desc = env.unwrapped.desc

      Q_values = []
      prev_q = 0
      for episode in range(episodes):
        state = env.reset()
        done = False
        t_reward = 0

        #Q_values.append( np.abs(np.sum(Q) - prev_q))

        Q_values.append( np.sum( Q[0,:]))
        prev_q = np.sum(Q)
        max_steps = 1000000
        #Q_prev = np.sum( Q)
        for i in range(max_steps):
          if done:
            break        
          current = state
          if np.random.rand()< (epsilon):
            action = np.argmax(Q[current, :])
          else:
            action = env.action_space.sample()
          
          state, reward, done, info = env.step(action)
          t_reward += reward
          Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])


        #epsilon*= epsilon_decay
        epsilon=(1-2.71**(-episode/1000))
        #print( epsilon)
        rewards.append(t_reward)
        iters.append(i)


      #breakpoint()
      for k in range(env.observation_space.n):
        optimal[k]=np.argmax(Q[k, :])



      games.append(run_episodes(  env, 0, optimal))


    plot_graph( alpha_arr, games, " reward vs alpha ", " alpha ", "games won")







      #error.append( np.sum( np.abs( Q - Q_prev)))




      

      #plot_graph( np.arange( 1, len( Q_values)+1), Q_values, " Q_values variation", "episodes", "Difference")







      #print( optimal)



    #display_value_iteration( np.array( optimal), env)


    #display_value_iteration( optimal, env)


    #plt.figure()
    #plt.grid()
    #plt.plot( np.arange( 1, episodes+1), error)


    #plt.title("Error values with iterations")
    #plt.xlabel(" iterations ")
    #plt.ylabel(" error values ")

    #plt.show()




breakpoint()

environment  = 'FrozenLake-v0'


random_map = generate_random_map(size=4, p=0.8)


env = gym.make("FrozenLake-v0")

env.render()
Q_learning_frozen_lake(env)
  

