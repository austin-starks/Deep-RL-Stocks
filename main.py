import os 
import logging
import datetime
import utils
from pathlib import Path
import train



if __name__ == "__main__":
    path = os.path.dirname(Path(__file__).absolute())
    # format_long = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    format_short = '[%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(
        filename=f'{path}/logs/{"log"}.log',
        format=format_short,
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO,
        filemode="w")

    policy, replay_buffer = train.run(["SPY"], '01-01-2009', '01-01-2015')
    train.test(["SPY"], '01-01-2016', '09-30-2018', policy, replay_buffer)


    