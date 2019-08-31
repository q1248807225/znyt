# coding:utf-8

import logging
import logging.config

logging.config.fileConfig("../conf/logger.ini")

def get_logger(name=None):
    return logging.getLogger(name)
