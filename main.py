import models.alternative.environment as env



if __name__ == "__main__":
    policy, replay_buffer = env.run(['SPY'], '01-01-2011', '01-01-2015', save_location="results/experiment2")
    env.test(['SPY'], '01-01-2016', '09-30-2018', policy, replay_buffer, save_location="results/experiment2")


    