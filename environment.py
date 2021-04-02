import pandas as pd
import re
import datetime

class Environment(object):
    """
    The current environment of the agent.

    The environment keeps track of where the agent is after taking action a in 
    state s.
    """

    def __init__(self, stock_name, start_date, end_date):
        """
        Initializes the environment.
        
        Parameter stock_name: the name of the stock for this environment.
        Precondition: stock_name must be a str of a stock (or etf)'s ticker

        Parameter start_date: the starting date of this environment.
        Precondition: start_date must be a string in this format: YYYY-DD-MM

        Parameter end_date: the ending date of this environment.
        Precondition: end_date must be a string in this format: YYYY-DD-MM and 
                    end_date must be after start_date
        """
        assert self._check_date_preconditions(start_date, end_date), "Date preconditions failed"
        self._stock_name = stock_name
        filename = "data/price_data/" + stock_name
        try:
            self._dataframe = pd.read_csv(filename, header=0, index_col="Date",
                                 names=["Date", "Open", "High", "Low", "Close", "Volume"])
        except:
            self._dataframe = None
            raise AssertionError(stock_name + " is not a stock or ETF.")
        self._start_date = start_date
        self._end_date = end_date
        self.state = None
    

    def step(self, action):
        """
        Takes action in the current state to get to the next state

        Returns an array [new_state, reward, done] where:
            - new_state (State object): state after taking action in the current state
            - reward (float): reward for taking action in the current state 
            - done (boolean): whether or not the run is done 
        """
        # state includes (stock_price, num_shares, buying_power)
        # reward = num_shares * change_in_price
        raise NotImplementedError()

    def reset(self):
        """
        Resets the environment to a random date in the first 33% of the range 
        with a random amount of positions and random amount of buying power
        """
        raise NotImplementedError()
    
    def _check_date_preconditions(self, start_date, end_date):
        """
        Returns: True if start_date and end_date are in the right format.
                False otherwise
        """
        try:
            start_arr = re.split(r'[\-]', start_date)
            end_arr = re.split(r'[\-]', end_date)
            date_is_valid = True
            for x, y in zip(start_arr, end_arr):
                date_is_valid = x.isdigit() and y.isdigit() and date_is_valid
                if date_is_valid:
                    date_is_valid = date_is_valid and int(x) > 0 and int(y) > 0
                else:
                    return date_is_valid
            date1 = [int(x) for x in re.split(r'[\-]', start_date)]
            date2 = [int(x) for x in re.split(r'[\-]', end_date)]
            date1_obj = datetime.date(date1[0], date1[1], date1[2])
            date2_obj = datetime.date(date2[0], date2[1], date2[2])
            epochs = (date2_obj - date1_obj).days
            return date_is_valid and epochs >= 0
        except:
            return False

