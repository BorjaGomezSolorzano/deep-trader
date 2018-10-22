# -*- coding: utf-8 -*-
"""
Created on 18/09/2018

@author: Borja
"""

import logging
import os

def get_logger(name):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../logs/deep-trader.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger