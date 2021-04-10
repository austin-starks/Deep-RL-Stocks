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

    train.run(stock_names=["F"], random_start=True)
    train.test(stock_names=["F"])


    