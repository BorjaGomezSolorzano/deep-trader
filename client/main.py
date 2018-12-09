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
from model.trainer import Model
from commons.interactive_plots import price_rewards_actions_utility_plot, convergence_plot
from commons.write_results import write, read_price_actions_rewards, read_convergence

logger = get_logger(os.path.basename(__file__))

def read_csv():

  dates, data, rewards, sharpe, decisions = read_price_actions_rewards()

  price_rewards_actions_utility_plot(True, dates, data, rewards, decisions, sharpe)

  rew_epochs = read_convergence()

  convergence_plot(rew_epochs)


def write_csv():

  feeder = Feeder()

  X, y, dates, instrument = feeder.process()

  model = Model()
  rewards, actions, dates_o, instrument_o, rew_epoch = model.execute(X, y, dates, instrument)

  write(dates_o, instrument_o, rewards, actions, rew_epoch)


if __name__ == '__main__':

  write_csv()

  read_csv()
