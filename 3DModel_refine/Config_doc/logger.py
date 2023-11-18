import logging
import sys

loggers = {}

'''
Prepare for the log function.
'''

def get_logger(name, level=logging.DEBUG):

    global loggers
    if loggers.get(name) is not None:
        return loggers[name] # return the logger information
    else:
        # create new logger
        logger = logging.getLogger(name)
        # Set the log level
        logger.setLevel(level)
        # Set the logger format

        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        # Output the log to a file
        file_handler = logging.FileHandler(f'{name}.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        file_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        # Save the log information to the document and console
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        loggers[name] = logger
        # return the new created log information
        return logger