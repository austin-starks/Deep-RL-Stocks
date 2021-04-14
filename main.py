import models.environment as env



if __name__ == "__main__":
    policy, replay_buffer = env.run(["AAPL", 'NVDA', 'MSFT'], '01-01-2011', '01-01-2015')
    env.test(["AAPL", 'NVDA', 'MSFT'], '01-01-2016', '09-30-2018', policy, replay_buffer)


    