import logging

def log_info(*msg):
    """
    Logs info to the console and to the logs

    This function prints the message and saves the message to the log
    """
    if len(msg) == 1:
        msg = msg[0]
    print(str(msg))
    logging.info(str(msg))