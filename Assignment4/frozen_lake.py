import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns


from gym.envs.toy_text.frozen_lake import generate_random_map







def plot_graph( x, y, title, x_label, y_label):
  plt.plot(x, y)
  plt.xlabel(x_label)
  plt.title(title)
  plt.ylabel(y_label)
  plt.grid()
  plt.show()






def error_analysis_frozen_lake( ):


  errors = []

  k_values = []




  for i in [4,6,8, 10,12]:
    environment  = 'FrozenLake-v0'

    random_map = generate_random_map(size=i, p=0.8)

    env = gym.make("FrozenLake-v0", desc=random_map)

    env.reset()

    env.render()

    env = env.unwrapped

    desc = env.unwrapped.desc

    best_policy,k, error = policy_iteration( env, gamma = 0.9)

    k_values.append(k+1)
    errors.append( error)



  plt.figure()
  plt.grid()
  plt.plot(np.arange( 1, k_values[0]), errors[0] ,label ='size =4')

  plt.plot(np.arange( 1, k_values[1]), errors[1], label ='size =6')

  plt.plot(np.arange( 1, k_values[2]), errors[2], label ='size =8')
  plt.plot(np.arange( 1, k_values[3]), errors[3], label ='size =10')
  plt.plot(np.arange( 1, k_values[4]), errors[4], label ='size =12')

  plt.ylabel('error value')
  plt.legend()
  plt.title('variation of error with iteration for different size')
  plt.show()





def error_analysis_frozen_lake_value( ):


  errors = []

  k_values = []




  for i in [4,6,8,10,12]:
    environment  = 'FrozenLake-v0'

    random_map = generate_random_map(size=i, p=0.8)

    env = gym.make("FrozenLake-v0", desc=random_map)

    env.reset()

    env.render()

    env = env.unwrapped

    desc = env.unwrapped.desc

    best_policy,k, error = value_iteration( env, gamma = 0.9)

    k_values.append(k+1)
    errors.append( error)



  plt.figure()
  plt.grid()
  plt.plot(np.arange( 1, k_values[0]), errors[0] ,label ='size =4')

  plt.plot(np.arange( 1, k_values[1]), errors[1], label ='size =6')

  plt.plot(np.arange( 1, k_values[2]), errors[2], label ='size =8')
  plt.plot(np.arange( 1, k_values[3]), errors[3], label ='size =10')
  plt.plot(np.arange( 1, k_values[4]), errors[4], label ='size =12')

  plt.ylabel('error value')
  plt.legend()
  plt.title('variation of error with iteration for different size')
  plt.show()






def iterations_analysis_frozen_lake_value( ):


  iterations = np.array( [0,0,0,0,0], dtype = np.float64)

  


  sizes = [4,6,8,10,12]




  for j in range( 30):

    print(j)
    k_values = []

    for i in [4,6,8,10,12]:
      environment  = 'FrozenLake-v0'

      random_map = generate_random_map(size=i, p=0.8)

      env = gym.make("FrozenLake-v0", desc=random_map)

      env.reset()

      #env.render()


      env = env.unwrapped

      desc = env.unwrapped.desc

      best_policy,k, error = value_iteration( env, gamma = 0.9)


      k_values.append(k+1)


    iterations+= np.array( k_values)

  iterations/=30





  plt.figure()
  plt.grid()
  plt.plot(sizes, iterations)

  plt.ylabel('iterations')
  plt.xlabel(' problem size')
  #plt.legend()
  plt.title('variation of error with iteration for different size')
  plt.show()


def threshold_variation_value_iteration(env):

  threshold_arr = [0.1,0.001,1e-5,1e-7,1e-9, 1e-11, 1e-13, 1e-15, 1e-17, 1e-19]
  v=[]
  iters = []

  games_won = []

  for i in threshold_arr: 

    value, k = value_iteration( env, 0.9, i)

    v.append( np.sum(value))
    policy = extract_policy(env,value, gamma = 0.9)

    games_won.append( run_episodes( env, value, policy))

    iters.append( k)


  plot_graph( threshold_arr, iters, "iters with threshold", "threshold", "iters")

  plot_graph( threshold_arr, v, " best value  with threshold", "threshold", "value")


  plot_graph( threshold_arr, games_won, " games won  with threshold", "threshold", "games")








# function for displaying a heatmap
def display_value_iteration(P, env ):
    nb_states = env.observation_space.n
    visited_states = np.zeros(nb_states).astype(bool)
    visited_states[0] = 1
    states_labels = np.where(P==0, '<', 
                              np.where(P==1, 'v', 
                                       np.where(P==2, '>', 
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
        states_labels[s] = '<' if best_a==0 else ('v' if best_a==1 else ('>' if best_a==2 else '^'))   
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
  epsilon_arr = [0.2,0.4,0.6,0.8,0.9,0.95]


  games = []


  
  for epsilon in [0.2,0.4,0.6,0.8, 0.9, 0.95]:

    error = []
    st = time.time()
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    iters = []
    optimal=[0]*env.observation_space.n
    alpha = 0.85
    gamma = 0.95
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
      epsilon=(1-2.71**(-episode/1000))
      #print( epsilon)
      rewards.append(t_reward)
      iters.append(i)







      #error.append( np.sum( np.abs( Q - Q_prev)))




    breakpoint()
    for k in range(env.observation_space.n):
      optimal[k]=np.argmax(Q[k, :])



    #print(epsilon)


    


    games.append(run_episodes(  env, 0, optimal))

  


    Q_values  = np.array(Q_values)
    Q_values/=1000



    #plot_graph( np.arange( 1, len( Q_values)+1), Q_values, " Q_values variation", "episodes", "Difference")


  plot_graph( epsilon_arr, games, "reward with epsilon", "epsilon", " games")







    


  



def Frozen_Lake_Experiments():
  # 0 = left; 1 = down; 2 = right;  3 = up




  environment  = 'FrozenLake-v0'
  

  random_map = generate_random_map(size=4, p=0.8)

  env = gym.make("FrozenLake-v0")


  env.reset()

  env.render()

  #env = gym.make(environment)
  env = env.unwrapped
  desc = env.unwrapped.desc

  print( env.render())

  time_array=[0]*10
  gamma_arr=[0]*10
  iters=[0]*10
  list_scores=[0]*10

  
  ### POLICY ITERATION ####
  print('POLICY ITERATION WITH FROZEN LAKE')
  for i in range(0,10):
    st=time.time()
    best_policy,k, error = policy_iteration(env, gamma = (i+0.5)/10)

    print( best_policy)
    #display_value_iteration( best_policy, env)
    scores = evaluate_policy(env, best_policy, gamma = (i+0.5)/10)
    end=time.time()
    gamma_arr[i]=(i+0.5)/10
    list_scores[i]=np.mean(scores)
    iters[i] = k
    time_array[i]=end-st



  title = "'Frozen Lake - Policy Iteration - Execution Time Analysis'"


  plot_graph( gamma_arr, time_array,title,"gamma", " execution time" )

  title = "Frozen Lake - Policy Iteration - Reward Analysis"

  plot_graph( gamma_arr, list_scores, title, "gamma", "reward")

  title = "Frozen Lake - Policy Iteration - Convergence Analysis"

  plot_graph( gamma_arr, iters, title, "gamma", "iterations")

  
  ### VALUE ITERATION ###
  print('VALUE ITERATION WITH FROZEN LAKE')
  best_vals=[0]*10
  for i in range(0,10):
    st=time.time()
    best_value,k= value_iteration(env, gamma = (i+0.5)/10)
    #display_value_iteration( best_value, env)
    policy = extract_policy(env,best_value, gamma = (i+0.5)/10)
    policy_score = evaluate_policy(env, policy, gamma=(i+0.5)/10, n=1000)
    gamma = (i+0.5)/10
    #plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Value Iteration) ' + 'Gamma: '+ str(gamma),policy.reshape(4,4),desc,colors_lake(),directions_lake())
    end=time.time()
    gamma_arr[i]=(i+0.5)/10
    iters[i]=k
    best_vals[i] = np.sum(best_value)
    list_scores[i]=np.mean(policy_score)
    time_array[i]=end-st


  plt.figure()




  title = "'Frozen Lake - Value Iteration - Execution Time Analysis'"


  plot_graph( gamma_arr, time_array,title,"gamma", " execution time(s)" )

  title = "Frozen Lake - Value Iteration - Reward Analysis"

  plot_graph( gamma_arr, list_scores, title, "gamma", "reward")

  title = "Frozen Lake - Value Iteration - Convergence Analysis"

  plot_graph( gamma_arr, iters, title, "gamma", "iterations")




  title = 'Frozen Lake - Value Iteration - Best Value Analysis'

  plot_graph( gamma_arr, best_vals, title, "gamma", 'optimal value')

  
  




def run_episode(env, policy, gamma, render = True):
  obs = env.reset()
  total_reward = 0
  step_idx = 0
  while True:
    if render:
      env.render()
    obs, reward, done , _ = env.step(int(policy[obs]))
    total_reward += (gamma ** step_idx * reward)
    step_idx += 1
    if done:
      break
  return total_reward

def evaluate_policy(env, policy, gamma , n = 10):
  scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
  return np.mean(scores)

def extract_policy(env,v, gamma):
  policy = np.zeros(env.nS)
  for s in range(env.nS):
    q_sa = np.zeros(env.nA)
    for a in range(env.nA):
      q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
    policy[s] = np.argmax(q_sa)
  return policy

def compute_policy_v(env, policy, gamma):
  v = np.zeros(env.nS)
  eps = 1e-5
  while True:
    prev_v = np.copy(v)
    for s in range(env.nS):
      policy_a = policy[s]
      v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
    if (np.sum((np.fabs(prev_v - v))) <= eps):
      break
  return v

def policy_iteration(env, gamma):
  policy = np.random.choice(env.nA, size=(env.nS))  
  max_iters = 200000
  error = []
  desc = env.unwrapped.desc
  for i in range(max_iters):
    old_policy_v = compute_policy_v(env, policy, gamma)
    new_policy = extract_policy(env,old_policy_v, gamma)
    error.append( np.sum (np.abs(policy-new_policy )))
    #if i % 2 == 0:
    # plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + 'Gamma: ' + str(gamma),new_policy.reshape(4,4),desc,colors_lake(),directions_lake())
    # a = 1
    if (np.all(policy == new_policy)):
      k=i+1
      break
    policy = new_policy


  #plt.plot(np.arange( 1, k+1),error)
  #plt.xlabel('Iterations')
  #plt.grid()
  #plt.title('Frozen Lake - Error analysis')
  #plt.ylabel('Error value')
  #plt.show()



  return policy,k, error












def value_iteration(env, gamma, threshold = 0.001):
  error = []
  v = np.zeros(env.nS)  # initialize value-function
  max_iters = 100000
  eps = threshold
  desc = env.unwrapped.desc
  for i in range(max_iters):
    prev_v = np.copy(v)
    for s in range(env.nS):
      q_sa = [sum([p*(r + gamma*prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
      v[s] = max(q_sa)
    #if i % 50 == 0:
    # plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Value Iteration) ' + 'Gamma: '+ str(gamma),v.reshape(4,4),desc,colors_lake(),directions_lake())

    #error.append( np.sum( np.abs( v - prev_v)))
    if (np.sum(np.fabs(prev_v - v)) <= eps):
      k=i+1
      break
  return v,k

def plot_policy_map(title, policy, map_desc, color_map, direction_map):
  fig = plt.figure()
  ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
  font_size = 'x-large'
  if policy.shape[1] > 16:
    font_size = 'small'
  plt.title(title)
  for i in range(policy.shape[0]):
    for j in range(policy.shape[1]):
      y = policy.shape[0] - i - 1
      x = j
      p = plt.Rectangle([x, y], 1, 1)
      p.set_facecolor(color_map[map_desc[i,j]])
      ax.add_patch(p)

      text = ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
               horizontalalignment='center', verticalalignment='center', color='w')
      

  plt.axis('off')
  plt.xlim((0, policy.shape[1]))
  plt.ylim((0, policy.shape[0]))
  plt.tight_layout()
  plt.savefig(title+str('.png'))
  plt.close()

  return (plt)




  





def Forest_Experiments():
  import mdptoolbox, mdptoolbox.example
  
  print('POLICY ITERATION WITH FOREST MANAGEMENT')
  P, R = mdptoolbox.example.forest(S = 2000)
  value_f = [0]*10
  policy = [0]*10
  iters = [0]*10
  time_array = [0]*10
  gamma_arr = [0] * 10
  for i in range(0,10):
    pi = mdptoolbox.mdp.PolicyIteration(P, R, (i+0.8)/10)
    pi.run()
    gamma_arr[i]=(i+0.5)/10
    value_f[i] = np.mean(pi.V)
    policy[i] = pi.policy
    iters[i] = pi.iter
    time_array[i] = pi.time


  plt.plot(gamma_arr, time_array)
  plt.xlabel('Gammas')
  plt.title('Forest Management - Policy Iteration - Execution Time Analysis')
  plt.ylabel('Execution Time (s)')
  plt.grid()
  plt.show()

  
  plt.plot(gamma_arr,value_f)
  plt.xlabel('Gammas')
  plt.ylabel('Average Rewards')
  plt.title('Forest Management - Policy Iteration - Reward Analysis')
  plt.grid()
  plt.show()

  plt.plot(gamma_arr,iters)
  plt.xlabel('Gammas')
  plt.ylabel('Iterations to Converge')
  plt.title('Forest Management - Policy Iteration - Convergence Analysis')
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
    pi = mdptoolbox.mdp.ValueIteration(P, R, (i+0.5)/10)
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
  
  print('Q LEARNING WITH FOREST MANAGEMENT')
  P, R = mdptoolbox.example.forest(S=2000,p=0.01)
  value_f = []
  policy = []
  iters = []
  time_array = []
  Q_table = []
  rew_array = []
  for epsilon in [0.05,0.15,0.25,0.5,0.75,0.95]:
    st = time.time()
    pi = mdptoolbox.mdp.QLearning(P,R,0.95, epsilon =epsilon, n_iter = 15000)
    end = time.time()
    pi.run()
    rew_array.append(pi.reward_array)
    value_f.append(np.mean(pi.V))
    policy.append(pi.policy)
    time_array.append(end-st)
    Q_table.append(pi.Q)
  
  plt.plot(range(0,10000), rew_array[0],label='epsilon=0.05')
  plt.plot(range(0,10000), rew_array[1],label='epsilon=0.15')
  plt.plot(range(0,10000), rew_array[2],label='epsilon=0.25')
  plt.plot(range(0,10000), rew_array[3],label='epsilon=0.50')
  plt.plot(range(0,10000), rew_array[4],label='epsilon=0.75')
  plt.plot(range(0,10000), rew_array[5],label='epsilon=0.95')
  plt.legend()
  plt.xlabel('Iterations')
  plt.grid()
  plt.title('Forest Management - Q Learning - Decaying Epsilon')
  plt.ylabel('Average Reward')
  plt.show()

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

  return

def colors_lake():
  return {
    b'S': 'green',
    b'F': 'skyblue',
    b'H': 'black',
    b'G': 'gold',
  }

def directions_lake():
  return {
    3: '⬆',
    2: '➡',
    1: '⬇',
    0: '⬅'
  }

def actions_taxi():
  return {
    0: '⬇',
    1:'⬆',
    2: '➡',
    3: '⬅',
    4: 'P',
    5: 'D'
  }

def colors_taxi():
  return {
    b'+': 'red',
    b'-': 'green',
    b'R': 'yellow',
    b'G': 'blue',
    b'Y': 'gold'
  }



breakpoint()

#policy_iteration()




environment  = 'FrozenLake-v0'


random_map = generate_random_map(size=4, p=0.8)


env = gym.make("FrozenLake-v0")

env.render()









Forest_Experiments()

#threshold_variation_value_iteration( env)

#Q_learning_forest_management()
#Q_learning_frozen_lake(env)

#error_analysis_frozen_lake_value()
#Frozen_Lake_Experiments()

#iterations_analysis_frozen_lake_value()

#Forest_Experiments()
#env = gym.make('FrozenLake8x8-v0')
#env.render()

#stateValues = value_iteration(env, max_iterations=100000)
#policy = get_policy(env, stateValues)
#get_score(env, policy,episodes=1000)
