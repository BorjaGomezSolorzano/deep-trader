# -*- coding: utf-8 -*-
"""
Created on 24/09/2018

@author: Borja
"""

import os
os.environ['TZ'] = 'utc'

import yaml
from feeder.price_feed import Feeder
from logUtils import get_logger
from model import Model

logger = get_logger(os.path.basename(__file__))

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../../config/config.yaml")

if __name__ == '__main__':

  config = yaml.load(open(filename, 'r'))

  feeder = Feeder(config['instrument'], config['features_idx'], config['instrument_idx'], config['trainPctg'])
  i_train, o_train, i_test, o_test = feeder.process()

  model = Model()
  model.train(i_train, o_train, i_test, o_test)