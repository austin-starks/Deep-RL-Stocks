from models.environment import *
import utility.utils as utils
import os.path
from pathlib import Path
import logging
import datetime
import random
from models.model import DDPG, ReplayBuffer
import sys
from torch.utils.tensorboard import SummaryWriter

NUMBER_OF_ITERATIONS = 1000000
MAX_LIMIT = 200
START_TIMESTEPS = 5000
BATCH_SIZE = 128
STD_GAUSSIAN_EXPLORATION_NOISE =  0.2

EPSILON = 1
EPSILON_DECR = 0.001
EPSILON_LOW = 0.025


def is_greedy(t):
    global EPSILON
    random_num = random.random()
    result = random_num < EPSILON
    EPSILON = max(EPSILON_LOW, EPSILON - EPSILON_DECR)
    return result


def select_action(env, state, policy, t):
    global STD_GAUSSIAN_EXPLORATION_NOISE
    if t < START_TIMESTEPS or is_greedy(t):
        action = env.action_space.sample()
    else:
        action = (
            policy.select_action(state.to_numpy())
            + np.random.normal(
                0,
                MAX_LIMIT * STD_GAUSSIAN_EXPLORATION_NOISE,
                size=(env.action_space.shape),
            )
        ).clip(-MAX_LIMIT, MAX_LIMIT)
        action = action.astype(np.int32)

    return action


def run(
    stock_names,
    start_date,
    end_date,
    random_start=True,
    save_location="results/initial_policy",
):
    env = StockEnv(
        stock_names,
        start_date,
        end_date,
        max_limit=MAX_LIMIT,
        random_start=random_start,
    )
    utils.log_info("Environment Initilized")
    writer = SummaryWriter()
    policy = DDPG(
        env.state.shape[0],
        env.action_space.shape[0],
        max_action=MAX_LIMIT,
        policy_freq=2,
        lr=2e-3,
    )
    # os.path.exists('initial_policy')
    if os.path.exists(save_location + "_actor"):
        print("Loaded policy")
        policy.load(save_location)
    replay_buffer = ReplayBuffer(env.state.shape[0], env.action_space.shape[0])
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    with tqdm(total=NUMBER_OF_ITERATIONS, file=sys.stdout) as pbar:
        for t in range(NUMBER_OF_ITERATIONS):
            episode_timesteps += 1
            # Select action randomly or according to policy
            action = select_action(env, state, policy, t)
            # Perform action
            next_state, reward, done = env.step(action.flatten())
            if pbar.n % 10 == 0:
                # utils.log_info(f"Date and Time: {env.get_date_and_time()}")
                # utils.log_info(f"Current Portfolio Value: {env.calculate_portfolio_value()}")
                pbar.set_description(
                    f"{env.get_date_and_time()[0]} | R: {reward} | A: {action} | H: {env.get_holdings()}"
                )
            if pbar.n % 200 == 0:
                policy.save(save_location)
            done_bool = float(done) if episode_timesteps < env.max_epochs else 0
            # Store data in replay buffer
            replay_buffer.add(
                state.to_numpy(), action, next_state.to_numpy(), reward, done_bool
            )
            if t >= START_TIMESTEPS:
                writer.add_scalars(
                    "holdings",
                    {sn: v for sn, v in zip(stock_names, env.get_holdings())},
                    t - START_TIMESTEPS,
                )
                writer.add_scalars(
                    "action",
                    {sn: v for sn, v in zip(stock_names, action)},
                    t - START_TIMESTEPS,
                )
                writer.add_scalar("reward", reward, t - START_TIMESTEPS)
            state = next_state
            episode_reward += reward
            # Train agent after collecting sufficient data
            if t >= START_TIMESTEPS:
                policy.train(replay_buffer, BATCH_SIZE)
            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # print(
                #     f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
                # )
                # Reset environment
                state, done = env.reset(), False
                utils.log_info("episode_reward", episode_reward)
                utils.log_info(
                    "reward per timestep", episode_reward / episode_timesteps
                )
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
            pbar.update()
    return policy, replay_buffer


def append_portfolio_value(df, env):
    value = env.calculate_portfolio_value()
    date, time = env.get_date_and_time()
    time = "09:30AM" if time == "Open" else "04:00PM"
    datetime = date + " " + time
    return df.append(
        pd.DataFrame([round(value, 2)], columns=["Portfolio Value"], index=[datetime])
    )


def test(
    stock_names,
    start_date,
    end_date,
    policy,
    replay_buffer,
    save_location="results/initial_policy",
):
    env = StockEnv(
        stock_names,
        start_date=start_date,
        end_date=end_date,
        max_limit=MAX_LIMIT,
        random_start=False
    )
    utils.log_info("Testing policy")
    state, done = env.reset(), False
    episode_reward = 0
    df = pd.DataFrame(columns=["Portfolio Value"])
    df = append_portfolio_value(df, env)
    utils.log_info("Testing...")
    while not done:
        # print(env.get_date_and_time())
        action = policy.select_action(state.to_numpy())
        utils.log_info(env.get_date_and_time(), "action", action)
        next_state, reward, done = env.step(action)
        done_bool = float(done)
        # replay_buffer.add(state.to_numpy(), action, next_state.to_numpy(), reward, done_bool)
        state = next_state
        episode_reward += reward
        # policy.train(replay_buffer, BATCH_SIZE)
        df = append_portfolio_value(df, env)
    df.to_csv(save_location, index_label="Date")



if __name__ == "__main__":
    # path = os.path.dirname(Path(__file__).absolute())
    # format_short = '[%(filename)s:%(lineno)d] %(message)s'
    # format_long = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    # logging.basicConfig(
    #     filename=f'{path}/logs/{str(datetime.datetime.now())}.log',
    #     format=format_long,
    #     datefmt='%Y-%m-%d:%H:%M:%S',
    #     level=logging.INFO,
    #     filemode="w")
    policy, replay_buffer = run(
        ["SPY", 'QQQ'],
        "01-01-2011",
        "01-01-2015",
        save_location="results/ddpg",
        random_start=False,
    )
    test(
        ["SPY"],
        "01-01-2016",
        "09-30-2018",
        policy,
        replay_buffer,
        save_location=f"results/test_results_ddpg_{NUMBER_OF_ITERATIONS}.csv",
    )

