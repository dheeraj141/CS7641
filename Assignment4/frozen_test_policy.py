

import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from gym.envs.toy_text.frozen_lake import generate_random_map


# using the Bellman equation, we find the action providing the highest value for the given state s. 
# V is the list of values of all states
def choose_best_action(env, V, s, gamma):
    a_best = None
    q_best = float('-inf')
    nb_actions = env.action_space.n
    for a in range (0, nb_actions):
        env.env.s = s # go to state s
        s_next, r, done, info = env.step(a) #take the action a
        q = r + gamma * V[s_next] # compute the value future value after taking action a
        if q > q_best:
            q_best = q
            a_best = a
    return a_best



# value iteration algorithm
def compute_value_iteration(env, 
                            gamma=.9, v_delta_threshold=.000001,
                            V = None, verbose=True):
    env.reset()
    nb_actions = env.action_space.n
    nb_states = env.observation_space.n
    # values vector
    if V == None:
        V = np.zeros([nb_states])
    # policy vector
    P = np.zeros([nb_states], dtype=int)
    iteration = 0
    while True:

        v_delta = 0
        for s in range (0, nb_states):
            v_previous = V[s]
            a_best = choose_best_action(env, V, s, gamma) # find an action with the highest future reward
            env.env.s = s # go to the state s
            s_next, r, done, info = env.step(a_best) #take the best action
            V[s] = r + gamma * V[s_next] # update the value of the state
            P[s] = a_best # store the best action in the policy vector for the state
            v_delta = max(v_delta, np.abs(v_previous - V[s])) # calculate the rate of value improvment for the state
        iteration += 1
        if v_delta < v_delta_threshold:
            if verbose:
                print (iteration,' iterations done')
            break
    return V, P

# compute values for a 4x4 board 
#V_4, P_4 = compute_value_iteration()
#print( V_4)



# function for displaying a heatmap
def display_value_iteration(P, env ):
    nb_states = env.observation_space.n
    visited_states = np.zeros(nb_states).astype(bool)
    visited_states[0] = 1
    states_labels = np.where(P==0, '<', 
                              np.where(P==1, '>', 
                                       np.where(P==2, 'v', 
                                                np.where(P==3, '^', P)
                                               )
                                      )
                             ) 
    desc = env.unwrapped.desc.ravel().astype(str)
    colors = np.where(desc=='S','y',np.where(desc=='F','b',np.where(desc=='H','r',np.where(desc=='G','g',desc))))
    states_labels = np.zeros(nb_states).astype(str)
    states_labels[:] = ''
    total_reward = 0
    s = env.reset()
    #env.render()
    done = False
    while done != True: 
        best_a = P[s] # select the best next action from the policy
        states_labels[s] = '^' if best_a==0 else ('v' if best_a==1 else ('>' if best_a==2 else '<'))   
        #print(s, best_a)
        s, rew, done, info = env.step(best_a) #take step using selected action
        total_reward = total_reward + rew
        visited_states[s] = 1 # mark the state as visited
        #env.render()
    ax = sns.heatmap(P.reshape(int(np.sqrt(nb_states)),int(np.sqrt(nb_states))), 
                 linewidth=0.5, 
                 annot=states_labels.reshape(int(np.sqrt(nb_states)),int(np.sqrt(nb_states))), 
                 cmap=list(colors),
                 fmt = '',
                 cbar=False)
    plt.show()
    print("Total Reward: ", total_reward)
    
# display heatmap for a 4x4 board
#display_value_iteration(P_4)





# function for performing policy iteration
def compute_policy_iteration(env, 
                            gamma=.9, v_delta_threshold=.00001,
                            P = None, verbose=True):
    env.reset()
    nb_actions = env.action_space.n
    nb_states = env.observation_space.n
    # values vector
    V = np.zeros([nb_states])
    # policy vector
    if P == None:
        P = np.random.choice(nb_actions, size=nb_states)
        P = np.zeros([nb_states], dtype=int)
        
    max_iterations = 200000
    iteration = 0
    for i in range(max_iterations):
        
        # policy evaluation
        while True:
            v_delta = 0
            for s in range (0, nb_states):
                v_previous = V[s]                
                env.env.s = s # go to state s
                s_next, r, done, info = env.step(P[s]) #take the action recommended by policy
                V[s] = r + gamma * V[s_next] # update value after applying policy
                v_delta = max(v_delta, np.abs(v_previous - V[s])) # calculate the rate of value improvment for the state
            if v_delta < v_delta_threshold:
                break
            print( V.reshape(4,4))

        # policy improvement
        policy_stable = True
        for s in range (0, nb_states):
            a_old = P[s] # ask policy for action to perform
            a_best = choose_best_action(env, V, s, gamma) # find an action with the highest future reward    
            P[s] = a_best # store the best action in the policy vector for the state
            if a_old != a_best:
                policy_stable = False
        
        if policy_stable:
            break
        print( P.reshape(4,4))
                
        iteration += 1
    if verbose:
        print (iteration,' iterations done')    
    return V, P
 








breakpoint()

env = gym.make('FrozenLake-v0', is_slippery=False)


Vp_4, Pp_4 = compute_policy_iteration(env)
print(Vp_4)
display_value_iteration(Pp_4, env)



# compute values for a 4x4 board 
V_4, P_4 = compute_value_iteration(env)
print( V_4)
display_value_iteration(P_4, env)




V_8, P_8 = compute_value_iteration(env = gym.make('FrozenLake8x8-v0', is_slippery=False))
print(V_8)
display_value_iteration(P_8, env = gym.make('FrozenLake8x8-v0', is_slippery=False))



V_8, P_8 = compute_policy_iteration(env = gym.make('FrozenLake8x8-v0', is_slippery=False))
print(V_8)
display_value_iteration(P_8, env = gym.make('FrozenLake8x8-v0', is_slippery=False))






env = gym.make('FrozenLake-v0')


Vp_4, Pp_4 = compute_policy_iteration(env)
print(Vp_4)
display_value_iteration(Pp_4, env)



# compute values for a 4x4 board 
V_4, P_4 = compute_value_iteration(env)
print( V_4)
display_value_iteration(P_4, env)




V_8, P_8 = compute_value_iteration(env = gym.make('FrozenLake8x8-v0'))
print(V_8)
display_value_iteration(P_8, env = gym.make('FrozenLake8x8-v0'))











