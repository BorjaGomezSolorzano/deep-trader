import sys
sys.path.append('../')

import os
os.environ['TZ'] = 'utc'

from model import *

logger = get_logger(os.path.basename(__file__))

def read_csv():

  dates, data, rewards, sharpe, decisions = read_price_actions_rewards()

  price_rewards_actions_utility_plot(True, dates, data, rewards, decisions, sharpe)

  rew_epochs = read_convergence()

  convergence_plot(rew_epochs)


def write_csv():

  X, y, dates, instrument = process()

  rewards, actions, dates_o, instrument_o, rew_epoch = execute(X, y, dates, instrument)

  write(dates_o, instrument_o, rewards, actions, rew_epoch)


if __name__ == '__main__':

  write_csv()

  read_csv()
