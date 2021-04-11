import train



if __name__ == "__main__":
    policy, replay_buffer = train.run(["AAPL", 'NVDA', 'MSFT'], '01-01-2009', '01-01-2015')
    train.test(["AAPL", 'NVDA', 'MSFT'], '01-01-2016', '09-30-2018', policy, replay_buffer)


    