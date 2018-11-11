# -*- coding: utf-8 -*-
"""
Created on 24/09/2018

@author: Borja
"""

import os
os.environ['TZ'] = 'utc'

import yaml
from commons.price_feed import Feeder
from commons.logUtils import get_logger
from model.model import Model
from commons.interactive_plots import plotly_interactive_decisions
from commons.write_results import write, read

logger = get_logger(os.path.basename(__file__))

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")

def read_csv():
  config = yaml.load(open(filename, 'r'))

  dates, data, rewards, sharpe, decisions = read(config)

  plotly_interactive_decisions(True, dates, data, rewards, decisions, sharpe, config=config)


def write_csv():
  config = yaml.load(open(filename, 'r'))

  feeder = Feeder(config)

  X, y, dates, instrument = feeder.process()

  model = Model()
  rewards, actions, dates_o, instrument_o = model.train(X, y, dates, instrument)

  write(config, dates_o, instrument_o, rewards, actions)


if __name__ == '__main__':

  write_csv()

  read_csv()
