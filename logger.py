import logging
import os


def setup_logger(name, file, level='info'):
    formatter = logging.Formatter('{asctime} {levelname} {message}', style='{')
    file_handler = logging.FileHandler(file, mode='w')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if level == 'info':
        logger.setLevel(logging.INFO)
    elif level == 'debug':
        logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


if not os.path.exists('./logs'):
    os.mkdir('./logs')
