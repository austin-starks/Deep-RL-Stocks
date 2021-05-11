import gym
from gym import spaces
import pandas as pd
import numpy as np
from tqdm import tqdm
from models.state import State
import random
import re
import datetime

class StockEnv(gym.Env):
    """
    The current environment of the agent.
    The environment keeps track of where the agent is after taking action a in
    state s.
    """

    def __init__(
        self,
        stock_names,
        start_date,
        end_date,
        max_limit,
        starting_amount_lower=20000,
        starting_amount_upper=50000,
        random_start=False,
        invalid_action_penalty=0
    ):
        """
        Initializes the environment.
        Parameter stock_names: the name of the stocks for this environment.
        Precondition: stock_names must be an array of stocks or ETFs
        Parameter start_date: the starting date of this environment.
        Precondition: start_date must be a string in this format: YYYY-DD-MM
        Parameter end_date: the ending date of this environment.
        Precondition: end_date must be a string in this format: YYYY-DD-MM and
                    end_date must be after start_date
        """
        super(StockEnv, self).__init__()
        self.random_start = random_start
        self.valid_dates = pd.read_csv("data/price_data/SPY.csv", index_col="Date").index
        self.number_of_stocks = len(stock_names)
        self.stock_names = stock_names
        self.initialize_date(start_date, end_date), "Date preconditions failed"
        self.starting_amount_lower = starting_amount_lower
        self.starting_amount_upper = starting_amount_upper
        self.starting_amount = self.starting_amount_upper
        self.reset(init=True)
        self.action_space = spaces.Box(
            low=-max_limit, high=max_limit, shape=(self.number_of_stocks,), dtype=np.int32
        )
        self.invalid_action_penalty = invalid_action_penalty

    def calculate_reward(self, holdings, remaining_money, stock_prices_new, action_is_invalid=False):
        buy_hold_comparison = (self.state.buy_hold_comparison * stock_prices_new).sum()
        buy_hold_last = self.buy_hold_last
        r = (
            remaining_money
            + np.sum(holdings * (stock_prices_new))
        )
        value_last = self.value_at_last_timestep if self.value_at_last_timestep != 0 else r

        self.value_at_last_timestep = r
        self.buy_hold_last = buy_hold_comparison
        if action_is_invalid:
            r = r - self.invalid_action_penalty # can penalize invalid actions
        return r - value_last

    def step(self, action):
        """
        Takes action in the current state to get to the next state
        Returns an array [new_state, reward, done] where:
            - new_state (State object): state after taking action in the current state
            - reward (float): reward for taking action in the current state
            - done (boolean): whether or not the run is done
        """
        current_date, current_time = self.get_date_and_time()
        stock_prices_old = self.state.get_stock_prices(current_date, current_time)
        # perform action: if buying, add positions. if selling, subtract positions.
        # change buying power
        holdings, remaining_money, action_is_invalid = self.state.get_new_holdings(action, stock_prices_old)
        self.increment_date()
        new_date, new_time = self.get_date_and_time()
        stock_prices_new = self.state.get_stock_prices(new_date, new_time)
        self.state.advance_state(remaining_money, holdings, new_date, new_time)
        reward = self.calculate_reward(holdings, remaining_money, stock_prices_new, action_is_invalid)
        return self.state, reward, self.is_done()

    def increment_date(self):
        """
        Increments the date by one epoch
        """
        incr = 1
        start_arr = list(map(lambda x: int(x), re.split(r"[\-]", self.start_date)))
        date_obj = datetime.date(start_arr[2], start_arr[0], start_arr[1])
        s = self.stock_names[0]
        adjusted_date = str(date_obj + datetime.timedelta((self.epochs + incr) // 2))
        while not (
            adjusted_date
            in self.valid_dates
        ):
            incr += 1
            adjusted_date = str(date_obj + datetime.timedelta((self.epochs + incr) // 2))
            if incr >= 20:
                raise Exception(f"{adjusted_date} is out of range")
        self.epochs += incr

    def is_done(self):
        """
        Returns: True if the episode is done. False otherwise
        """
        return self.epochs >= self.max_epochs

    def reset(self, init=False):
        """
        Resets the environment to a random date in the first 33% of the range
        with a random amount of positions and random amount of buying power
        """
        if self.random_start:
            starting_money = [random.randint(
                self.starting_amount_lower, self.starting_amount_upper
            )]
            starting_shares = [
                random.randint(0, 10) for _ in range(self.number_of_stocks)
            ]
        else:
            starting_money = [self.starting_amount_upper]
            starting_shares = [0 for _ in range(self.number_of_stocks)]
        starting_money = np.array(starting_money)
        starting_shares = np.array(starting_shares)
        self.value_at_last_timestep = 0
        self.initialize_starting_epoch(self.start_date, self.end_date)
        current_date, current_time = self.get_date_and_time()
        if init:
            self.state = State(self.stock_names, starting_money, starting_shares, current_date, current_time)
        else:
            self.state.reset(starting_money, starting_shares, current_date, current_time)
        self.starting_amount = self.state.calculate_portfolio_value()
        self.buy_hold_last = self.state.buy_hold_comparison
        return self.state

    def get_date_and_time(self):
        """
        Gets current date and time
        """
        time = "Open" if self.epochs % 2 == 0 else "Close"
        start_arr = list(map(lambda x: int(x), re.split(r"[\-]", self.start_date)))
        date_obj = datetime.date(
            start_arr[2], start_arr[0], start_arr[1]
        ) + datetime.timedelta(self.epochs // 2)
        return str(date_obj), time

    def calculate_portfolio_value(self):
        """
        Calculates the current portfolio value
        """
        return self.state.calculate_portfolio_value()

    def get_holdings(self):
        """
        Returns: the current holdings
        """
        return self.state.get_holdings()

    def initialize_date(self, start_date, end_date):
        """
        Returns: True if start_date and end_date are in the right format.
                False otherwise
        """
        start_arr = re.split(r"[\-]", start_date)
        end_arr = re.split(r"[\-]", end_date)
        date_is_valid = True
        for x, y in zip(start_arr, end_arr):
            date_is_valid = x.isdigit() and y.isdigit() and date_is_valid
            if date_is_valid:
                date_is_valid = date_is_valid and int(x) > 0 and int(y) > 0
            else:
                return date_is_valid
        date1 = [int(x) for x in re.split(r"[\-]", start_date)]
        date2 = [int(x) for x in re.split(r"[\-]", end_date)]
        date1_obj = datetime.date(date1[2], date1[0], date1[1])
        date2_obj = datetime.date(date2[2], date2[0], date2[1])
        epochs = (date2_obj - date1_obj).days
        if not (date_is_valid and epochs >= 0):
            raise ValueError("Date is not valid")
        self.max_epochs = epochs * 2
        self.start_date = start_date
        self.end_date = end_date

    def initialize_starting_epoch(self, start_date, end_date):
        """
        Gets the starting epoch of a cycle
        """
        if self.random_start:
            date1 = [int(x) for x in re.split(r"[\-]", start_date)]
            date2 = [int(x) for x in re.split(r"[\-]", end_date)]
            date1_obj = datetime.date(date1[2], date1[0], date1[1])
            date2_obj = datetime.date(date2[2], date2[0], date2[1])
            self.epochs = random.randint(-1, int((date2_obj - date1_obj).days * 0.2))
        else:
            self.epochs = -1
        self.increment_date()  # needed to be sure we're not on a weekend/holiday





