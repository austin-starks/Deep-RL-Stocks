
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
        stock_prices = self.get_stock_prices(current_date, current_time)
        self.essential_state = np.concatenate([
            starting_money, starting_shares, stock_prices
        ])
        self.past_state = PastState(len(self.essential_state), days_in_state)
        self.past_state.add(self.essential_state)
        self.get_indicators()
        self.indicator_state = self.get_indicator_state(current_date, current_time)
        state1, state2 = self.get_state()
        self.shape = (state1.shape, state2.shape)
        self.buy_hold_comparison = self.calculate_portfolio_value() / self.number_of_stocks / stock_prices
      
    
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
        stock_prices = self.get_stock_prices(current_date, current_time)
        self.essential_state = np.concatenate([
            starting_money, starting_shares, stock_prices
        ])
        self.buy_hold_comparison = self.calculate_portfolio_value() / self.number_of_stocks / stock_prices
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
        