# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:19:10 2024

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

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3) 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x #F.softmax(x, dim=1)

#policy_net = torch.load('policy_network_complete_final.pth')
policy_net = PolicyNetwork()
policy_net.load_state_dict(torch.load('policy_network_all_models_episode400_10firm.pth'))
policy_net.eval()  # Set the model to evaluation mode
"""
apple_data = pd.read_excel("Risk_Analysis_Data_Adjusted.xlsx", sheet_name="AAPL", )
apple_data.set_index('Unnamed: 0', inplace=True)
apple_data = apple_data.iloc[:365,:]

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
"""

def draw_actions(action_list,apple_closing_prices):
    action_list = np.asarray(action_list)
    initial_cash = 10000
    holdings = 0
    plt.figure()
    num_1 = 0
    num_2 = 0
    plt.plot(np.arange(0,len(apple_closing_prices)),apple_closing_prices)
    plt.plot(np.where(action_list==1)[0],apple_closing_prices[np.where(action_list==1)[0]],"g.")
    plt.plot(np.where(action_list==2)[0],apple_closing_prices[np.where(action_list==2)[0]],"r.")
    plt.plot(np.where(action_list==0)[0],apple_closing_prices[np.where(action_list==0)[0]],"b.")    
    for i,action in enumerate(action_list):
        
        if action == 1:
            initial_cash -= 10*apple_closing_prices[i]
            holdings +=10
            num_1+=1
        elif action == 2:
            initial_cash += 10*apple_closing_prices[i]
            holdings -=10
            num_2 += 1
            
    plt.title(f"AAPL Data result 03/04/2023 - 03/04/2024\ninitial_cash = {initial_cash:.2f} --- holdings = {holdings} --- holdings price = {(holdings*apple_closing_prices[-1]):.2f}")
    plt.xlabel("Days")
    plt.ylabel("Amount of $ gain")
    plt.legend(["Closing Prices", "Buy", "Sell","Hold"])
    return num_1, num_2
    
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


symbols_train = ["AAPL", "CRVL", "EGHT", "IBM", "INTC", "MSFT", "NSIT",
           "PRO", "RMBS","ADBE"]

symbols_test = ["ORCL", "SAP", "MU", "SONY", "CTSH", "ANSS", "HPQ",
           "CSCO"]

def performance(sheet, symbols):
    best_results = {}
    best_actions = {}
    averages = {}
    best_res = 0
    avg_res = 0
    for symbol in symbols:
        best_res = 0
        avg_res = 0
        for i in range(20):
            apple_data = pd.read_excel(sheet, sheet_name=symbol )
            apple_data.set_index('Unnamed: 0', inplace=True)
            apple_data = apple_data.iloc[:365,:]
            
            #apple_data = apple_data.drop(apple_data.columns[0], axis=1).dropna()
            apple_data = apple_data.apply(lambda col: col[::-1]) # Reverse for chronological order
            
            apple_closing_prices = np.asarray(apple_data['Close'])
            
            scaler = StandardScaler()
            
            # Fit scaler on the data and transform the data
            df_normalized = scaler.fit_transform(apple_data)
            
            # Convert the result back to a DataFrame
            apple_data_normalized = pd.DataFrame(df_normalized, columns=apple_data.columns)
            apple_data_normalized = apple_data_normalized[["DebtToEquity","ADX 14 Period","RSI 14 Period","Crossover Signal"]]
            
            
            env = Env(apple_data_normalized, apple_closing_prices)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            env.reset()
            
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
                action_list.append(action.item())
    
            
                state = next_state
                steps +=1
            
                
            best_reward = env.cash + env.close_price*env.holdings
            if best_reward > best_res:
                best_res = best_reward
                best_actions[symbol] = action_list
                best_results[symbol] = best_reward
            avg_res += best_reward 
    
            #draw_actions(action_list,apple_closing_prices)
            print(f'Company {symbol}, Cash: {env.cash}, Holdings: {env.close_price*env.holdings}')
        averages[symbol] = avg_res/20
        print(f'Company {symbol}, Average: {avg_res/20}')
    return best_results, best_actions, averages

best_results_train, best_actions_train, averages_train = performance("Risk_Analysis_Data_Adjusted.xlsx",symbols_train)
best_results_test, best_actions_test, averages_test = performance("Risk_Analysis_Data_test.xlsx",symbols_test)

plt.figure()
plt.bar(range(len(averages_train)), list(best_results_train.values()), align='center')
plt.bar(range(len(averages_train)), list(averages_train.values()), align='center')
plt.plot(range(len(averages_train)), np.ones(len(averages_train))*10000, "r-")
plt.xticks(range(len(averages_train)), list(averages_train.keys()))        
plt.ylabel("Amount of $ gain")
plt.legend(["Intial amount", "Maximum Gain", "Average Gain"])
plt.title("REINFORCE Result for Train Data 03/04/2023 - 03/04/2024")

plt.figure()
plt.bar(range(len(averages_test)), list(best_results_test.values()), align='center')
plt.bar(range(len(averages_test)), list(averages_test.values()), align='center')
plt.plot(range(len(averages_test)), np.ones(len(averages_test))*10000, "r-")
plt.xticks(range(len(averages_test)), list(averages_test.keys()))        
plt.ylabel("Amount of $ gain")
plt.legend(["Intial amount", "Maximum Gain", "Average Gain"])
plt.title("REINFORCE Result for Test Data 03/04/2023 - 03/04/2024")

