import os 
import logging
import datetime
import utils
from pathlib import Path


if __name__ == "__main__":
    path = os.path.dirname(Path(__file__).absolute())

    # Can log with utils.log_info(args)
    logging.basicConfig(
        filename=f'{path}/logs/{datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")}.log',
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)