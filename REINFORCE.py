# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 08:21:39 2024

@author: esola
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
%matplotlib qt

symbols = ["AAPL", "IBM", "INTC","ADBE", "MSFT"]
data = {}
close_price = {}
weights = {}

for i in symbols:

    apple_data = pd.read_excel("Risk_Analysis_Data_Adjusted.xlsx", sheet_name=i )
    apple_data.set_index('Unnamed: 0', inplace=True)
    apple_data = apple_data.iloc[:2700,:]
    
    #apple_data = apple_data.drop(apple_data.columns[0], axis=1).dropna()
    apple_data = apple_data.apply(lambda col: col[::-1]) # Reverse for chronological order
    
    apple_closing_prices = np.asarray(apple_data['Close'])
    
    scaler = StandardScaler()
    
    # Fit scaler on the data and transform the data
    df_normalized = scaler.fit_transform(apple_data)
    
    # Convert the result back to a DataFrame
    apple_data_normalized = pd.DataFrame(df_normalized, columns=apple_data.columns)
    #apple_data_normalized = apple_data_normalized.drop(['Close'],axis=1)
    apple_data_normalized = apple_data_normalized[["DebtToEquity","ADX 14 Period","RSI 14 Period","Crossover Signal"]]
    apple_data_normalized = apple_data_normalized.set_index(apple_data.index)
    data[i] = apple_data_normalized
    close_price[i] = apple_closing_prices
    weights[i] = 1/len(symbols)
    

def update_weights(weights, selected_symbol):
    weights[selected_symbol] *= 0.9
    w = np.asarray(list(weights.values()))
    w /= w.sum()
    for i,sym in enumerate(weights.keys()):
        weights[sym] = w[i]
    
    return weights

def select_train_set(data, close_price, weights):
    selected_symbol = np.random.choice(list(weights.keys()), p=list(weights.values()))
    selected_date = np.random.randint(0, 2100)
    
    weights = update_weights(weights, selected_symbol)
    
    selected_data = data[selected_symbol].iloc[selected_date:selected_date+260*2,:]
    selected_close_price = close_price[selected_symbol][selected_date:selected_date+260*2]
    
    
    return selected_data, selected_close_price, weights, selected_symbol







def draw_actions(action_list,apple_closing_prices):
    initial_cash = 10000
    holdings = 0
    plt.figure()
    num_1 = 0
    num_2 = 0
    plt.plot(np.arange(0,len(apple_closing_prices)),apple_closing_prices)
    for i,action in enumerate(action_list):
        if action == 1:
            plt.plot(i,apple_closing_prices[i],"g.")
            initial_cash -= 10*apple_closing_prices[i]
            holdings +=10
            num_1+=1
        elif action == 2:
            plt.plot(i,apple_closing_prices[i],"r.")
            initial_cash += 10*apple_closing_prices[i]
            holdings -=10
            num_2 += 1
        else:
            plt.plot(i,apple_closing_prices[i],"b.")
    plt.title("initial_cash = " + str(initial_cash) + " --- " + "holdings = " + str(holdings) + " --- " + "holdings price = " + str(holdings*apple_closing_prices[-1]))
    return num_1, num_2

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x #F.softmax(x, dim=1)


                

class Env:

    def __init__(self, data, close_price_data, initial_cash=10000, transaction_cost=0, penalty=100):
        self.data = data
        self.close_price_data = close_price_data
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost  
        self.penalty = penalty  
        self.amount = 10
        self.reset()
        
        
    def reset(self):
        self.current_step = 0
        self.close_price = self.close_price_data[self.current_step]
        self.cash = self.initial_cash
        self.holdings = 0  
        self.total_asset_value = self.cash
        self.last_total_asset_value = self.cash
        self.done = False
        self.num_of_hold_action = 0
        self.amount = 10
        return self._next_observation()
    
    def _next_observation(self):
        state = np.array(self.data.iloc[self.current_step][:])
        return state
    
    def step(self,action):
        self.close_price = self.close_price_data[self.current_step]
        if action == 1:
            self.cash -= self.amount*self.close_price
            self.holdings += self.amount
        if action == 2:
            self.cash += self.amount*self.close_price
            self.holdings -= self.amount
        if action == 0:
            self.cash= self.cash
            self.holdings = self.holdings
            self.num_of_hold_action +=1
            
        self.total_asset_value = self.cash + self.holdings*self.close_price
        if (self.total_asset_value > self.initial_cash):
            reward = 1
        else: 
            reward = 0
        #reward = self.total_asset_value - self.last_total_asset_value
        
        self.current_step +=1
        if self.current_step >= len(self.close_price_data)-1:
            self.done = True
        if self.holdings < self.amount and self.cash < self.close_price_data[self.current_step]*self.amount:
            self.done = True
        if self.total_asset_value < self.initial_cash*0.8:
            self.done = True
        return self._next_observation(), reward, self.done, {}
    
    def get_action_mask(self):

        if self.holdings < self.amount and self.cash < self.close_price_data[self.current_step]*self.amount:
            return torch.tensor([0.0, float('-inf'), float('-inf')])
        elif self.holdings < self.amount:
            return torch.tensor([0.0, 0.0, float('-inf')])
        elif self.cash < self.close_price_data[self.current_step]*self.amount:

            return torch.tensor([0.0, float('-inf'),0.0])
        elif self.num_of_hold_action >= 10:
            self.num_of_hold_action = 0
            return torch.tensor([float('-inf'),0.0,0.0])
        else:
            return torch.tensor([0.0,0.0,0.0])
        
    
    
    
        
        
            
        
   
policy_net = PolicyNetwork()

# REINFORCE Algorithm Parameters
learning_rate = 0.001
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)


acts = {}

def train_reinforce(env, policy_net, optimizer, episodes=400):
    acts = {}
    best_reward = 0
    best_action_list = list()
    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        steps = 1
        action_list = []

        # Generate an episode
        while not done:
            state = torch.FloatTensor([state]).to(device)
            probs = policy_net(state)
            action_mask = env.get_action_mask()
            
            masked_actions = action_mask.to(probs.device) + probs
            masked_probs = F.softmax(masked_actions, dim=-1)
            
            m = Categorical(masked_probs)
            action = m.sample()
            next_state, reward, done,k= env.step(action.item())
            action_list.append(action)
            log_probs.append(m.log_prob(action))
            rewards.append(reward)

            state = next_state
            steps +=1

        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + 0.99 * R  # Discount factor gamma = 0.99
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Policy gradient update
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        if env.cash + env.close_price*env.holdings > best_reward:
            best_reward = env.cash + env.close_price*env.holdings
            best_action_list = action_list
        #acts[episode] = (action_list,env.cash,env.holdings)
        print(f'Episode {episode}, Loss: {policy_loss.item()}, Reward: {sum(rewards)}, Cash: {env.cash}, Holdings: {env.close_price*env.holdings}')
        #if episode % 2 == 0:
        #    print(env.cash)
        #    print(env.close_price*env.holdings)


        #    print(f'Episode {episode}, Loss: {policy_loss.item()}')
        #    print(f'Episode {episode}, Reward: {sum(rewards)}')
            #print(f'initial cash {initial_cash}, holdings {holdings}')
    return acts, policy_net

# Run the training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net.to(device)
for i in range(10):
    selected_data, selected_close_price, weights, selected_symbol = select_train_set(data, close_price, weights)
    env = Env(selected_data, selected_close_price)
    print("Train for: " + selected_symbol + " between " + selected_data.index[0] + " - " + selected_data.index[-1])
    acts, policy_net = train_reinforce(env, policy_net, optimizer)


    




