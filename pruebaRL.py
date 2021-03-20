import numpy as np      # Library required for matrix creation  and manipulation
import math
import time             # Library required to calculate execution time
import pandas as pd     # Library required to manipulate Dataframes with the model results

np.set_printoptions(precision=2, suppress=True)

cost_inventory = 2     # The cost of holding inventory
purchase_price = 20    # The price at which the inventory is bought
sales_price = 50       # The proce at which the inventory is sold

max_state = 800
initial_state = np.random.randint(max_state)

R = np.matrix(np.zeros([max_state,max_state]))

for y in range(0, max_state):
    for x in range(0, max_state):
        R[x,y] = np.maximum((x-y)*sales_price,0)-np.maximum((y-x)*purchase_price,0)-x*cost_inventory
        
print('R: \n', R)

Q = np.matrix(np.random.random([max_state,max_state]))

learning_rate = 0.5
discount = 0.7
EPISODES =3000
STEPS = 200
PRINT_EVERY = EPISODES/50
Pepsilon_init = 0.8   # initial value for the decayed-epsilon-greedy method
Pepsilon_end = 0.1    # final value for the decayed-epsilon-greedy method

def available_actions(state, customers):
    # The available actions are 
    #    a.- Meeting a customer requirement (going to s: state-order)
    #    b.- Buying Inventory from Supplier (going to s: state+purchase)
    purchase = np.arange(state, max_state) # Calculate all possible future states due to purchases from the current state
    # print('Purchase: ',purchase)
    new_customers_state =[]
    new_customers_state = [np.maximum(state-x,0) for x in customers] # calculate the possible states from customers in the current state
    # print('new_customers_state: ', new_customers_state)
    return np.concatenate((purchase,new_customers_state))

def sample_next_action(available_act, epsilon):
    # here we choose the type of next action to take. 1 for a random next action with probability epsilon 
    # and 0 for a greedy next action with probability (1-epsilon)
    random_action = np.random.binomial(n=1, p=epsilon, size=1)
    
    if random_action == 1:
        # This is the option for full exploration - always random
        # print('random action')
        next_action = int(np.random.choice(available_act, 1))
    else:
        # This is the option for full exploitation - always use what we know (Greedy method)
        # Choose the next actions from the available actions, and the one with the highest Q Value
        # print('greedy action')
        next_action = np.int(np.where(Q[current_state,] == np.max(Q[current_state,available_act]))[1]) 
    # This section just caluclates the amount that is being sold or purchsed, if at all
    if next_action < current_state:
        Qsale = current_state-next_action
        Qpurchase = 0
    else: 
        Qpurchase = next_action - current_state
        Qsale = 0
    return next_action, Qsale, Qpurchase

def cost_inventory_backlog(current_state):
    if current_state<=0:
        return cost_backlog
    else:
        return cost_inventory
    
# this function updates the Q matrix according to the path selected and the Q learning algorithm
def update(current_state, action):
   
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1] # index for the maximum Q value in the future_state
    # print('Q[action,]: \n', Q[action,])
    # print('Current State: ', current_state)
    # print('Action: ', action)
    # print('Max Index:', max_index)
    
    # just in case there are more than one maximums, in which case we choose one at random
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index] # this is the maximum Q valuein the future state given the action that generates that maximum value
    
    # Q learning formula
    Q[current_state, action] = (1-learning_rate)*Q[current_state, action] + learning_rate*(R[current_state, action] + discount*max_value) 
    
# Training

start_time = time.time()
epsilon = Pepsilon_init
epsilon_delta = (Pepsilon_init - Pepsilon_end)/EPISODES

calculation_times = []
total_reward = []
total_demand =[]

total_jump = []
jump_max = []
jump_min = []
jump_av =[]
jump_sd = []

total_state= []
state_max = []
state_min = []
state_av =[]
state_sd = []

current_state = 0

for episode in range(EPISODES):
    # Initialize values for Step
    total_reward_episode = 0
    start_time_episode = time.time()
    
    state_episode = []
    jump_episode = []

    # determine if this eisode id generating a status output
    if episode%PRINT_EVERY == 0:
        print('Calc. Episode {} of {}, {:.1f}% progress'.format(episode, EPISODES, 100*(episode/EPISODES)))
    # Execute the steps in the Episode    
    for step in range(STEPS):
        # Create a customer for this step
        customers = []
        customers.append(np.random.randint(0,max_state))
        # Calculate the actions (future states) that are available from current state
        available_act = available_actions(current_state, customers)
        # Choose an action from the available future states
        action = sample_next_action(available_act, epsilon)
        # Update the Q table 
        update(current_state, action[0])

        # record the states for the step
        # av_demand_episode =
        total_state.append(current_state)
        state_episode.append(current_state)
        total_demand.append(customers[0])
        total_reward_episode += R[current_state, action[0]]
        total_jump.append(action[0] - current_state)
        jump_episode.append(action[0] - current_state)
        # update the state for the next step
        current_state = action[0]

    # record the states for the Episode
    total_reward.append(total_reward_episode) # Total reward for the episode
    calculation_times.append(time.time()-start_time_episode)
    
    jump_max.append(np.max(jump_episode))
    jump_min.append(np.min(jump_episode))
    jump_av.append(np.mean(jump_episode))
    jump_sd.append(np.std(jump_episode))
    
    state_max.append(np.max(state_episode))
    state_min.append(np.min(state_episode))
    state_av.append(np.mean(state_episode))
    state_sd.append(np.std(state_episode))

    # Update parameters for the next episode
    epsilon = Pepsilon_init - episode*epsilon_delta
    current_state = np.random.randint(0, int(Q.shape[0]))

# print out the total calculation time
print('total calculation time: {:.2f} seconds'.format(time.time()-start_time))

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
%matplotlib inline

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))

MA_total_reward = pd.DataFrame(total_reward)

Rolling_total_reward = MA_total_reward.rolling(window=5).mean()

axs[0,0].plot(MA_total_reward, label='Episode')
axs[0,0].plot(Rolling_total_reward, color='r', label ='MA(5)')
axs[0,0].set_title('Total Rewards')
axs[0,0].set_ylabel('Rewards [$]')
axs[0,0].set_xlabel('Episodes')
axs[0,0].grid(axis='y', alpha=0.75)
axs[0,0].grid(axis='x', alpha=0.75)

axs[1,0].plot(calculation_times)
axs[1,0].set_title('Calc. Times')
axs[1,0].set_xlabel('Episodes')
axs[1,0].set_ylabel('Calculation times [s]')
axs[1,0].grid(axis='y', alpha=0.75)
axs[1,0].grid(axis='x', alpha=0.75)

axs[0,1].hist(total_state,color='#0504aa',alpha=0.7, rwidth=0.85)
axs[0,1].set_title('States Histogram')
axs[0,1].set_xlabel('State')
axs[0,1].set_ylabel('Frequency')
axs[0,1].set_xlim(xmin=0, xmax=max_state)
axs[0,1].grid(axis='y', alpha=0.75)

axs[1,1].plot(jump_max,color='b', label = 'max')
axs[1,1].plot(jump_min,color='r', label = 'min')
axs[1,1].plot(jump_av,color='g', label = 'av')
axs[1,1].plot(jump_sd,color='y', label = 'sd')
axs[1,1].set_title('Jumps')
axs[1,1].legend()
axs[1,1].set_xlabel('Episode')
axs[1,1].set_ylabel('Jump Value')
axs[1,1].grid(axis='y', alpha=0.75)

axs[0,2].hist(total_jump,color='#0504aa',alpha=0.7, rwidth=0.85)
axs[0,2].set_title('Jump Histogram')
axs[0,2].set_xlabel('New_State-Old_State')
axs[0,2].set_ylabel('Frequency')
axs[0,2].set_xlim(xmin=-max_state, xmax=max_state)
axs[0,2].grid(axis='y', alpha=0.75)

axs[1,2].plot(state_max,color='b', label = 'max')
axs[1,2].plot(state_min,color='r', label = 'min')
axs[1,2].plot(state_av,color='g', label = 'av')
axs[1,2].plot(state_sd,color='y', label = 'sd')
axs[1,2].set_title('States')
axs[1,2].legend()
axs[1,2].set_xlabel('Episode')
axs[1,2].set_ylabel('State Value')
axs[1,2].grid(axis='y', alpha=0.75)


plt.tight_layout()
plt.show()
