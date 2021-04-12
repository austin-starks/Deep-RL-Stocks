import pandas as pd
import numpy as np

class State(object):
    """
    Represents the internal state of an environment
    """
    def __init__(self, stock_names, starting_money, starting_shares, current_date, current_time):
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
        self.state = np.concatenate([
            starting_money, starting_shares, self.get_stock_prices(current_date, current_time)
        ])
        self.shape = self.state.shape
    
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
        Gets the new holdingd after taking action
        """
        old_holdings = self.state[1 : 1 + self.number_of_stocks]
        current_cash = self.state[0]
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
        return self.state[1:1+self.number_of_stocks]
    
    def calculate_portfolio_value(self):
        """
        Calculates the current portfolio value
        """
        return self.state[0] + np.sum(
            self.state[1 : 1 + self.number_of_stocks]
            * self.state[1 + self.number_of_stocks : 1 + 2 * self.number_of_stocks]
        )
    
    def advance_state(self, remaining_money, holdings, stock_prices):
        self.state = np.concatenate([
            np.array([remaining_money]), holdings, stock_prices
        ])  
    
    def to_numpy(self):
        return self.state.copy()