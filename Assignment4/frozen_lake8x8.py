

import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt


from gym.envs.toy_text.frozen_lake import generate_random_map







s =4
environment  = 'FrozenLake-v0'

random_map = generate_random_map(size=s, p=0.4)


env = gym.make("FrozenLake-v0", desc=random_map)

env.reset()

env.render()


env = env.unwrapped
nA = env.action_space.n
nS = env.observation_space.n
V = np.zeros(nS)


def policy_evaluation(V, policy, eps=0.0001):
    while True:
        delta = 0
        for s in range(nS):
            old_v = V[s]
            V[s] = eval_state_action(V, s, policy[s])
            delta = max(delta, np.abs(old_v - V[s]))
        if delta < eps:
            break


def policy_improvement(V, policy):
    policy_stable = True
    for s in range(nS):
        old_a = policy[s]
        policy[s] = np.argmax([eval_state_action(V, s, a) for a in range(nA)])
        if old_a != policy[s]: 
            policy_stable = False
    return policy_stable









def eval_state_action(V, s, a, gamma=0.99):
    return np.sum([p * (rew + gamma*V[next_s]) for p, next_s, rew, _ in env.P[s][a]])


def policy_iteration():
  
  policy = np.zeros(nS)

  policy_stable = False
  it = 0
  while not policy_stable:
    policy_evaluation(V, policy)
    policy_stable = policy_improvement(V, policy)
    it += 1
  print('Converged after %i policy iterations'%(it))
  run_episodes(env, V, policy)
  print(V.reshape((s,s)))

  print(policy.reshape((s,s)))




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






policy_iteration()









