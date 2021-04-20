Based on this paper: https://arxiv.org/pdf/2103.11455.pdf

Highlights:
    - Critic uses LSTM layers. (Previously attempted Actor uses it too, but WAYY too slow)
    - Epsilon-Greedy Strategy


State: 

import pandas as pd
import numpy as np
import datetime
import os.path
from pathlib import Path


class State(object):
    """
    Represents the internal state of an environment
    """
    def __init__(self, stock_names, starting_money, starting_shares, current_date, current_time, days_in_state=100):
        """
        Initializes the State of the environment

        Parameter stock_name: the name of the stocks for the state.
        Precondition: stock_names must be an array of stocks or ETFs

        Parameter starting_money: the initial amount of buying power for the state
        Precondition: starting_money must be an array of buying power or ETFs

        Parameter starting_shares: the initial amount of shares for the state
        Precondition: starting_shares must be an array of stocks or ETFs

        Parameter current_date: the current date of the state.
        Precondition: current_date must be a string in this format: YYYY-DD-MM

        Parameter current_time: the current time of the state.
        Precondition: current_time must be a string in this format: HH:MM
        """
        self.dataframes = dict()
        self.stock_names = stock_names
        self.number_of_stocks = len(stock_names)
        self.days_in_state = days_in_state
        
        if type(stock_names) == str:
            stock_names = [stock_names]
        for stock_name in stock_names:
            path = os.path.dirname(Path(__file__).absolute())
            filename = f"{path}/../data/price_data/{stock_name}.csv"
            try:
                self.dataframes[stock_name] = pd.read_csv(filename, index_col="Date")
            except:
                raise AssertionError(stock_name + " is not a stock or ETF.")
        self.essential_state = np.concatenate([
            starting_money, starting_shares, self.get_stock_prices(current_date, current_time)
        ])
        self.past_state = PastState(len(self.essential_state), days_in_state)
        self.past_state.add(self.essential_state)
        self.get_indicators()
        self.indicator_state = self.get_indicator_state(current_date, current_time)
        state1, state2 = self.get_state()
        self.shape = (state1.shape, state2.shape)
      
    
    def get_indicator_state(self, current_date, current_time):
        """
        Returns: The past 'days' of the indicator state
        """
        date_arr = [int(x) for x in current_date.split('-')]
        date_obj = datetime.date(date_arr[0], date_arr[1], date_arr[2]) - datetime.timedelta(self.days_in_state)
        past_date = str(date_obj)
        result = []
        for stock in self.stock_names:
            data = self.dataframes[stock].copy().loc[past_date: current_date]

            if current_time == 'Open':
                # We do not know the High, Low, Close, or indicators of the current date at open 
                # We must zero them out so the agent is not looking at the future
                open_price = data.loc[current_date]['Open']
                data.loc[current_date] = 0
                data.loc[current_date]['Open'] = open_price
            # print("data", data)
            data_as_numpy = data.to_numpy()        
            result.append(data_as_numpy)

        return np.array(result)

    def get_stock_prices(self, current_date, current_time):
        """
        Gets the current stock price at this epoch
        """
        result = []
        for stock in self.stock_names:
            price = self.dataframes[stock].loc[current_date][current_time]
            result.append(price)
        return np.array(result)
    
    def get_new_holdings(self, action, stock_prices):
        """
        Returns: the new holdings after performing action in the current state
        """
        old_holdings = self.essential_state[1 : 1 + self.number_of_stocks]
        current_cash = self.essential_state[0]
        new_holdings = []
        invalid_action = False
        for a, holding, price in zip(action, old_holdings, stock_prices):
            if a > 0:
                cash = a * current_cash / len(old_holdings)
                shares = cash / price
                total_price = shares * price
                current_cash -= total_price 
                new_holdings.append(holding + shares)
                
            else:
                shares = abs(a) * holding 
                total_price = shares * price
                current_cash += total_price 
                new_holdings.append(holding - shares)
        return np.array(new_holdings), current_cash, invalid_action
    
    def get_holdings(self):
        """
        Returns: the current holdings
        """
        return self.essential_state[1:1+self.number_of_stocks]
    
    def calculate_portfolio_value(self):
        """
        Returns: the current portfolio value
        """
        return self.essential_state[0] + np.sum(
            self.essential_state[1 : 1 + self.number_of_stocks]
            * self.essential_state[1 + self.number_of_stocks : 1 + 2 * self.number_of_stocks]
        )
    
    def advance_state(self, remaining_money, holdings, current_date, current_time):
        """
        Advances the state to the next state

        Parameter remaining_money (int): The buing power in the new state
        Parameter holdings (int[]): The holdings of each stock in the state
        Parameter current_date (string): The date of the new state
        Parameter current_time (string): The time of the new state
        """
        if current_time == 'Close':
            self.past_state.add(self.essential_state)
        stock_prices = self.get_stock_prices(current_date, current_time)
        self.essential_state = np.concatenate([
            np.array([remaining_money]), holdings, stock_prices
        ])  
        self.indicator_state = self.get_indicator_state(current_date, current_time)


    
    def get_indicators(self):
        """
        Adds indicators to the dataframe
        """
        for stock in self.stock_names:
            # get MACD
            df = self.dataframes[stock]
            exp1 = df.ewm(span=12, adjust=False).mean()
            exp2 = df.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            df['macd'] = macd['Close']

            # get moving averages
            df["seven_day_mean_moving_average"] = df.rolling(window=7).mean()['Close']
            df["thirty_day_mean_moving_average"] = df.rolling(window=30).mean()['Close']
            df["ninety_day_mean_moving_average"] = df.rolling(window=90).mean()['Close']
            df["two_hundred_day_mean_moving_average"] = df.rolling(window=200).mean()['Close']

            df["seven_day_std_moving_average"] = df.rolling(window=7).std()['Close']
            df["thirty_day_std_moving_average"] = df.rolling(window=30).std()['Close']
            df["ninety_day_std_moving_average"] = df.rolling(window=90).std()['Close']
            df["two_hundred_day_std_moving_average"] = df.rolling(window=200).std()['Close']

            # get bollander bands
            df["upper_bolliander_band"] = df.rolling(window=20).mean()['Close'] + 2 * df.rolling(window=20).std()['Close']
            df["lower_bolliander_band"] = df.rolling(window=20).mean()['Close'] - 2 * df.rolling(window=20).std()['Close']
            
            # get rsi
            diff = df['Close'].diff(1).dropna()       
            up_chg = 0 * diff
            down_chg = 0 * diff
            
            up_chg[diff > 0] = diff[ diff>0 ]            
            down_chg[diff < 0] = diff[ diff < 0 ]

            up_chg_avg   = up_chg.ewm(com=13 , min_periods=14).mean()
            down_chg_avg = down_chg.ewm(com=13 , min_periods=14).mean()
            
            rs = abs(up_chg_avg/down_chg_avg)
            rsi = 100 - 100/(1+rs)
            df['rsi'] = rsi
            self.dataframes[stock] = df.dropna()
            self.dataframes[stock]
        

    
    def reset(self, starting_money, starting_shares, current_date, current_time):
        """
        Resets the state with the new parameters
        """
        self.essential_state = np.concatenate([
            starting_money, starting_shares, self.get_stock_prices(current_date, current_time)
        ])
        self.past_state.reset()
        self.past_state.add(self.essential_state)
    
    def to_numpy(self):
        """
        Returns the numpy array representing the state object

        Alias for self.get_state()
        """
        state= self.get_state()
        return state

    def get_state(self):
        """
        Returns: the internal array representing the state
        """
        num_stocks, length, num_indicators = self.indicator_state.shape
        reshaped_indicator_state = self.indicator_state.reshape((length, num_stocks * num_indicators))
        length = len(reshaped_indicator_state)
        reshaped_indicator_state = reshaped_indicator_state[length - int(0.6 * self.days_in_state):length]
        return reshaped_indicator_state, self.past_state



class PastState(object):
    """
    Represents the past state of State
    """
    def __init__(self, days_in_state, max_size):
        """
        Initializes the past state
        """
        self.max_size = max_size
        self.days_in_state = days_in_state
        self.reset()
    
    def __len__(self):
        """
        Returns: The length of the state
        """
        return len(self.data)
    
    def __getitem__(self, *args):
        """
        Returns: get item of the past state
        """
        return self.data.__getitem__(args)

    def reset(self):
        """
        Resets the state to the initial state
        """
        self.data = np.zeros((self.max_size, self.days_in_state))
        self.current_size = 0
        self.shape = self.data.shape
        
    
    def add(self, essential_state):
        """
        Adds the state to the past state queue
        """
        if self.current_size < self.max_size:
            self.data[self.max_size - self.current_size - 1] = essential_state
            self.current_size += 1
        else:
            self.data = np.vstack((essential_state, self.data[:-1]))
    
    def copy(self):
        """
        Returns a copy of the internal state
        """
        return self.data.copy()
        

model:

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utility.utils as utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# Original Implementation found on https://github.com/sfujim/TD3/blob/master/TD3.py
class CNN(nn.Module):
    def __init__(self, indicator_state_dim, immediate_state_dim, hidden_size, outchannel, activation=nn.ReLU):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(indicator_state_dim[1], hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            activation(),
            nn.Conv1d(hidden_size, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm1d(outchannel),
        )
        self.relu = activation()
        if indicator_state_dim[1] == outchannel:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(indicator_state_dim[1], outchannel, kernel_size=1)
        
        self.output = nn.Linear(outchannel * indicator_state_dim[0], outchannel)
    
    def forward(self, X, X_immediate):
        out =  self.layers(X.permute(0,2,1)).permute(0,2,1)
        shortcut = self.shortcut(X.permute(0,2,1)).permute(0,2,1)
        out = self.relu(out + shortcut)
        shape = out.shape
        out = out.reshape((shape[0], shape[1] * shape[2]))
        out = self.output(out)   
        return out

class Actor(nn.Module):
    def __init__(self, ind_state_dim, imm_state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.conv = CNN(ind_state_dim, imm_state_dim, 64, 64)
        self.l1 = nn.Linear(64 , 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

        self.max_action = max_action


    def forward(self, ind_state, imm_state):
        ind_state = self.conv(ind_state, imm_state)
        a = F.relu(self.l1(ind_state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        return a

class RNN(nn.Module):
    def __init__(self, indicator_state_dim, immediate_state_dim, hidden_size, outchannel, activation=nn.ReLU):
        super(RNN, self).__init__()
        
        self.lstm1 = nn.LSTM(indicator_state_dim[0], hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.35)
        self.lstm2 = nn.LSTM(immediate_state_dim[0], hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.35)
        self.output = nn.Linear(2 * hidden_size * indicator_state_dim[1] + 2 * hidden_size * immediate_state_dim[1], outchannel)
    
    def forward(self, X, X_immediate):
        out = self.lstm1(X.permute(0,2,1))[0].permute(0,2,1)
        out2 = self.lstm2(X_immediate.permute(0,2,1))[0].permute(0,2,1)
   
        shape = out.shape
        out = out.reshape((shape[0], shape[1] * shape[2]))
        
        shape2 = out2.shape 
        out2 = out2.reshape((shape2[0], shape2[1] * shape2[2]))

        concat = self.output(torch.cat((out, out2), 1))   
        return concat


class Actor(nn.Module):
    def __init__(self, ind_state_dim, imm_state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.conv = CNN(ind_state_dim, imm_state_dim, 64, 64)
        self.l1 = nn.Linear(64 , 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

        self.max_action = max_action


    def forward(self, ind_state, imm_state):
        ind_state = self.conv(ind_state, imm_state)
        a = F.relu(self.l1(ind_state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        return a


class Critic(nn.Module):
    def __init__(self, indicator_state_dim, immediate_state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.rnn = RNN(indicator_state_dim, immediate_state_dim, 64, 64)
        self.l1 = nn.Linear(64 + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

        # Q2 architecture
        self.rnn2 = RNN(indicator_state_dim, immediate_state_dim, 64, 64)
        self.l4 = nn.Linear(64 + action_dim, 64)
        self.l5 = nn.Linear(64, 64)
        self.l6 = nn.Linear(64, 1)

    def forward(self, indicator_state, immediate_state, action):
        sa1 = self.rnn(indicator_state, immediate_state)
        sa1 = torch.cat([sa1, action], 1)
        q1 = F.relu(self.l1(sa1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        sa2 = self.rnn2(indicator_state, immediate_state)
        sa2 = torch.cat([sa2, action], 1)
        q2 = F.relu(self.l4(sa2))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, indicator_state_dim, immediate_state, action):
        sa = self.rnn(indicator_state_dim, immediate_state)
        sa = torch.cat([sa, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr=3e-4
    ):
        indicator_state_dim, immediate_state_dim = state_dim
        self.actor = Actor(indicator_state_dim, immediate_state_dim[0], action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(indicator_state_dim, immediate_state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_loss = torch.nn.SmoothL1Loss()

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state_tup):
        ind_state, imm_state = state_tup
        if (len(ind_state.shape) == 2):
            ind_state = torch.FloatTensor([ind_state]).to(device)
            imm_state = torch.FloatTensor([imm_state]).to(device)
        else:
            ind_state = torch.FloatTensor(ind_state).to(device)
            imm_state = torch.FloatTensor(imm_state).to(device)
        action = self.actor(ind_state, imm_state).cpu().data.numpy()
        return action

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        ind_state, imm_state, action, next_ind_state, next_imm_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy 
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_ind_state, next_imm_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_ind_state, next_imm_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(next_ind_state, next_imm_state, action)

        # Compute critic loss
        critic_loss = self.critic_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(next_ind_state, next_imm_state, 
                                self.actor(next_ind_state, next_imm_state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer")
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        indicator_state, immediate_state = state_dim
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        if type(indicator_state) == int:
            full_indicator_state = [max_size] + [indicator_state]
        else:
            full_indicator_state = [max_size] + [s for s in indicator_state]
        if type(immediate_state) == int:
            full_immediate_state = [max_size] + [immediate_state]
        else:
            full_immediate_state = [max_size] + [s for s in immediate_state]
        self.action = np.zeros((max_size, action_dim))
        self.indicator_state = np.zeros(full_indicator_state)
        self.immediate_state = np.zeros(full_immediate_state)
        self.next_indicator_state = np.zeros(full_indicator_state)
        self.next_immediate_state = np.zeros(full_immediate_state)
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, 
            state, 
            action, 
            next_state,
            reward, 
            done):
        indicator_state, immediate_state = state 
        next_indicator_state, next_immediate_state = next_state 

        self.immediate_state[self.ptr] = immediate_state
        self.indicator_state[self.ptr] = indicator_state
        self.action[self.ptr] = action
        self.next_immediate_state[self.ptr] = next_immediate_state
        self.next_indicator_state[self.ptr] = next_indicator_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.indicator_state[ind]).to(self.device),
            torch.FloatTensor(self.immediate_state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_indicator_state[ind]).to(self.device),
            torch.FloatTensor(self.next_immediate_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

