import pandas as pd
import numpy as np

class PastState(object):
    """
    Represents the past state of State
    """
    def __init__(self, state_length, max_size=156):
        """
        Initializes the past state
        """
        self.max_size = max_size
        self.state_length = state_length
        self.reset()
    
    def reset(self):
        """
        Resets the state to the initial state
        """
        self.data = np.zeros((self.max_size, self.state_length))
        self.current_size = 0
        self.shape = self.data.shape
        
    
    def add(self, essential_state):
        """
        Adds the state to the past state queue
        """
        if self.current_size < self.max_size:
            self.data[self.current_size] = essential_state
            self.current_size += 1
        else:
            self.data = np.vstack((essential_state, self.data[:-1]))
        





class State(object):
    """
    Represents the internal state of an environment
    """
    def __init__(self, stock_names, starting_money, starting_shares, current_date, current_time):
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
        if type(stock_names) == str:
            stock_names = [stock_names]
        for stock_name in stock_names:
            filename = f"data/price_data/{stock_name}.csv"
            try:
                self.dataframes[stock_name] = pd.read_csv(filename, index_col="Date")
            except:
                raise AssertionError(stock_name + " is not a stock or ETF.")
        self.essential_state = np.concatenate([
            starting_money, starting_shares, self.get_stock_prices(current_date, current_time)
        ])
        self.past_state = PastState(len(self.essential_state))
        self.past_state.add(self.essential_state)
        self.shape = self.essential_state.shape
        self.get_indicators()
      
    
    def __len__(self):
        """
        Returns: The length of the state
        """
        return len(self.past_state)

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
        for a, holding, price in zip(action, old_holdings, stock_prices):
            if current_cash - (a * price) >= 0 and holding + a >= 0:
                new_holdings.append(max(0, holding + a))
                current_cash -= a * price
            else:
                new_holdings.append(holding)
        return np.array(new_holdings), current_cash
    
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
        self.past_state.add(self.essential_state)
        stock_prices = self.get_stock_prices(current_date, current_time)
        self.essential_state = np.concatenate([
            np.array([remaining_money]), holdings, stock_prices
        ])  
    
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
            df["bolliander_band"] = df.rolling(window=20).mean()['Close'] + 2 * df.rolling(window=20).std()['Close']
            
            diff = df['Close'].diff(1).dropna()       
            up_chg = 0 * diff
            down_chg = 0 * diff
            
            up_chg[diff > 0] = diff[ diff>0 ]            
            down_chg[diff < 0] = diff[ diff < 0 ]
            
            # check pandas documentation for ewm
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
            # values are related to exponential decay
            # we set com=time_window-1 so we get decay alpha=1/time_window
            up_chg_avg   = up_chg.ewm(com=13 , min_periods=14).mean()
            down_chg_avg = down_chg.ewm(com=13 , min_periods=14).mean()
            
            rs = abs(up_chg_avg/down_chg_avg)
            rsi = 100 - 100/(1+rs)
            df['rsi'] = rsi
            self.dataframes[stock] = self.dataframes[stock].dropna()






    
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
        """
        return self.essential_state