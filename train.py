import gym
from gym import spaces
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import TD3, ReplayBuffer

NUMBER_OF_ITERATIONS = 50000
MAX_LIMIT = 10
START_TIMESTEPS = 2500
BATCH_SIZE = 128
STD_GAUSSIAN_EXPLORATION_NOISE = 0.1


class StockEnv(gym.Env):
    def __init__(self, csv_files, starting_money=50000):
        super(StockEnv, self).__init__()
        self.starting_money = starting_money
        dfs = []
        for csv in csv_files:
            dfs.append(pd.read_csv(csv))
        self.stock_prices = pd.concat(dfs, axis=1)
        self.stock_prices = self.stock_prices[
            ["Open", "Close", "High", "Low"]
        ].to_numpy()
        self.number_of_stocks = len(dfs)
        self.day = 0
        # size = 1 (money) + total feature size + holdings
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(1 + self.stock_prices.shape[1] + self.number_of_stocks,),
        )
        self.state = (
            [starting_money] + self.stock_prices[self.day] + [0] * self.number_of_stocks
        )

        self.action_space = spaces.Box(
            low=-MAX_LIMIT, high=MAX_LIMIT, shape=(self.number_of_stocks,), dtype=np.int
        )

    def _calculate_reward(self, actions):
        r = 0
        return r

    def step(self, action):
        reward = self._calculate_reward(action)
        remaining_money = 0
        holdings = [0] * self.number_of_stocks

        
        self.day += 1
        self.state = [remaining_money] + self.stock_prices[self.day] + holdings
        return self.state, reward, self.day == self.stock_prices.shape[0], {}

    def reset(self):
        self.day = 0
        self.state = (
            [self.starting_money]
            + self.stock_prices[self.day]
            + [0] * self.number_of_stocks
        )
        return self.state

    def render(self, mode="human"):
        return self.state


env = StockEnv(
    ["data/price_data/AAPL.csv", "data/price_data/ABC.csv", "data/price_data/AMD.csv"]
)
policy = TD3(env.state.shape[0], env.action_space.shape[0], max_action=MAX_LIMIT)
replay_buffer = ReplayBuffer(env.state.shape[0], env.action_space.shape[0])

state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0

with tqdm(total=NUMBER_OF_ITERATIONS) as pbar:
    for t in range(NUMBER_OF_ITERATIONS):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < START_TIMESTEPS:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(
                    0,
                    MAX_LIMIT * STD_GAUSSIAN_EXPLORATION_NOISE,
                    size=env.action_space.shape[0],
                )
            ).clip(-MAX_LIMIT, MAX_LIMIT)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= START_TIMESTEPS:
            policy.train(replay_buffer, BATCH_SIZE)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

